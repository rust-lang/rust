
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Write as _};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::mem;

use regex::Regex;

struct Os {
    pub name: &'static str,
    pub src: &'static str,
    pub fn_start: &'static str,
    pub scope_retxt: &'static str,
}

fn some_codes(dir: &str, Os { src, fn_start, scope_retxt,.. }: &Os)
              -> HashMap<String, Vec<String>>
{
    let src = format!("{}/{}", dir, src);
    let f = File::open(&src).expect(&format!("failed to open {}", &src));
    let f = BufReader::new(f);
    let mut f = f.lines().zip(1..);

    'fn_start: loop {
        for (l,_lno) in &mut f {
            let l = l.expect("read line");
            if l.starts_with(fn_start) { break 'fn_start }
        }
        panic!("failed to find {:?} in {}", fn_start, src);
    };

    let ignore_re = Regex::new(r#"^ *(?:use |match|//| _ *=> | +\}|  } *| *$)"#).unwrap();
    let ent_retxt = format!(r#"(:?x +== +)?{}\w+\b"#, scope_retxt);
    let lhs_retxt = format!(r#"^ *(:?x +if|\|)? +(?P<v>{}(?: *\|\|? *{})*)"#, &ent_retxt, &ent_retxt);
    let part_re = Regex::new(&format!(r#"({}) *\|? *$"#, &lhs_retxt)).unwrap();
    let full_retxt = format!(r#"({}) *=> *(?:return)? *(?P<k>\w+),? *$"#, &lhs_retxt);
    let full_re = Regex::new(&full_retxt).unwrap();
    let vprefix_re = Regex::new(&format!(r#"^{}"#, scope_retxt)).unwrap();

    let mut current = String::new();
    let mut out = HashMap::new();
    let more_current = |c: &mut String, m: &regex::Captures| {
        *c += " | ";
        *c += m.name("v").unwrap().as_str();
    };

    for (l, lno) in &mut f {
        let l = l.expect("read line");
        if l.trim_end() == "}" {
            break
        } else if ignore_re.is_match(&l) {
            // ok
        } else if let Some(m) = part_re.captures(&l) {
            more_current(&mut current, &m);
        } else if let Some(m) = full_re.captures(&l) {
            more_current(&mut current, &m);
            let key = m.name("k").unwrap().as_str();
            for e in mem::take(&mut current).split('|') {
                let e = e.trim();
                if e.len() == 0 { continue }
                if let Some(m) = vprefix_re.find(e) {
                    let v = &e[m.end()..];
                    out.entry(key.to_owned())
                        .or_insert(vec![])
                        .push(v.to_owned())
                }
            }
        } else {
            panic!("{}:{}: uncategorisable line {:?}", src, lno, &l);
        }
    }
    out
}

static OSLIST: &[Os] = &[
    Os {
        name: "Unix",
        src: "sys/unix/mod.rs",
        fn_start: "pub fn decode_error_kind(",
        scope_retxt: r#"libc::"#,
    },
    Os {
        name: "Windows",
        src: "sys/windows/mod.rs",
        fn_start: "pub fn decode_error_kind(",
        scope_retxt: r#"c::"#,
    },
];

static HEADING: &str = "/// # OS error codes";

fn main() {
    let dir = "library/std/src/";
    let osses = OSLIST.iter().map(|os| (os.name, some_codes(dir, os))).collect::<Vec<_>>();
    eprintln!("{:#?}", &osses);

    let src = format!("{}/io/error.rs", dir);
    let orig = fs::read_to_string(&src).expect(&format!("failed to open {}", &src));
    let mut upd: Vec<Cow<str>> = orig.split_inclusive('\n').map(Into::into).collect::<Vec<_>>();

    let mut i = 0;
    loop {
        let l = &upd[i];
        if l.starts_with("pub enum ErrorKind {") { break }
        i += 1;
    }

    let entry_re = Regex::new(r#"^ *(?P<k>\w+),\n$"#).unwrap();

    for i in i.. {
        let l = &upd[i];
        if l.starts_with("}") { break }

        let k = match entry_re.captures(&l) {
            Some(m) => m.name("k").unwrap(),
            None => continue,
        }.as_str().to_owned();

        let mut j = i;
        loop {
            let l = &upd[j];
            if l.trim_start().starts_with("///") { break }
            j -= 1;
        }
        let last_doc = j;
        let add_after = loop {
            let l = &upd[j];
            if l.trim_start().starts_with(HEADING) {
                if upd[j-1].trim() == "///" { j -= 1 }
                for l in &mut upd[j..=last_doc] { *l = "".into() }
                break j;
            }
            if ! l.trim_start().starts_with("///") {
                break last_doc;
            }
            j -= 1;
        };

        if osses.iter().any(|(_name,codes)| codes.get(&k).is_some()) {
            let addto = upd[add_after].to_mut();
            write!(addto, "    ///\n    {}\n", HEADING).unwrap();
            for (name,codes) in &osses {
                write!(addto, "    ///\n    /// {}:", name).unwrap();
                match codes.get(&k) {
                    Some(codes) => for code in codes {
                        write!(addto, " `{}`", code).unwrap();
                    }
                    None => {
                        write!(addto, " none").unwrap();
                    }
                }
                write!(addto, ".\n").unwrap();
            }
        }
    }

    let new = upd.join("");

    if new != orig {
        let dst = format!("{}.tmp", &src);
        let mut f = File::create(&dst).expect(&format!("create {}", &dst));
        write!(f, "{}", &new).unwrap();
        f.flush().unwrap();
        fs::rename(&dst, &src).expect(&format!("install {} as {}", &dst, &src));
    }
}

