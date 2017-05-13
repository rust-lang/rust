use std::fmt;

use Table as TomlTable;
use Value::{self, String, Integer, Float, Boolean, Datetime, Array, Table};

struct Printer<'a, 'b:'a> {
    output: &'a mut fmt::Formatter<'b>,
    stack: Vec<&'a str>,
}

struct Key<'a>(&'a [&'a str]);

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            String(ref s) => write_str(f, s),
            Integer(i) => write!(f, "{}", i),
            Float(fp) => {
                try!(write!(f, "{}", fp));
                if fp % 1.0 == 0.0 { try!(write!(f, ".0")) }
                Ok(())
            }
            Boolean(b) => write!(f, "{}", b),
            Datetime(ref s) => write!(f, "{}", s),
            Table(ref t) => {
                let mut p = Printer { output: f, stack: Vec::new() };
                p.print(t)
            }
            Array(ref a) => {
                try!(write!(f, "["));
                for (i, v) in a.iter().enumerate() {
                    if i != 0 { try!(write!(f, ", ")); }
                    try!(write!(f, "{}", v));
                }
                write!(f, "]")
            }
        }
    }
}

fn write_str(f: &mut fmt::Formatter, s: &str) -> fmt::Result {
    try!(write!(f, "\""));
    for ch in s.chars() {
        match ch {
            '\u{8}' => try!(write!(f, "\\b")),
            '\u{9}' => try!(write!(f, "\\t")),
            '\u{a}' => try!(write!(f, "\\n")),
            '\u{c}' => try!(write!(f, "\\f")),
            '\u{d}' => try!(write!(f, "\\r")),
            '\u{22}' => try!(write!(f, "\\\"")),
            '\u{5c}' => try!(write!(f, "\\\\")),
            ch => try!(write!(f, "{}", ch)),
        }
    }
    write!(f, "\"")
}

impl<'a, 'b> Printer<'a, 'b> {
    fn print(&mut self, table: &'a TomlTable) -> fmt::Result {
        for (k, v) in table.iter() {
            match *v {
                Table(..) => continue,
                Array(ref a) => {
                    if let Some(&Table(..)) = a.first() {
                        continue;
                    }
                }
                _ => {}
            }
            try!(writeln!(self.output, "{} = {}", Key(&[k]), v));
        }
        for (k, v) in table.iter() {
            match *v {
                Table(ref inner) => {
                    self.stack.push(k);
                    try!(writeln!(self.output, "\n[{}]", Key(&self.stack)));
                    try!(self.print(inner));
                    self.stack.pop();
                }
                Array(ref inner) => {
                    match inner.first() {
                        Some(&Table(..)) => {}
                        _ => continue
                    }
                    self.stack.push(k);
                    for inner in inner.iter() {
                        try!(writeln!(self.output, "\n[[{}]]", Key(&self.stack)));
                        match *inner {
                            Table(ref inner) => try!(self.print(inner)),
                            _ => panic!("non-heterogeneous toml array"),
                        }
                    }
                    self.stack.pop();
                }
                _ => {},
            }
        }
        Ok(())
    }
}

impl<'a> fmt::Display for Key<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, part) in self.0.iter().enumerate() {
            if i != 0 { try!(write!(f, ".")); }
            let ok = part.chars().all(|c| {
                match c {
                    'a' ... 'z' |
                    'A' ... 'Z' |
                    '0' ... '9' |
                    '-' | '_' => true,
                    _ => false,
                }
            });
            if ok {
                try!(write!(f, "{}", part));
            } else {
                try!(write_str(f, part));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(warnings)]
mod tests {
    use Value;
    use Value::{String, Integer, Float, Boolean, Datetime, Array, Table};
    use std::collections::BTreeMap;

    macro_rules! map( ($($k:expr => $v:expr),*) => ({
        let mut _m = BTreeMap::new();
        $(_m.insert($k.to_string(), $v);)*
        _m
    }) );

    #[test]
    fn simple_show() {
        assert_eq!(String("foo".to_string()).to_string(),
                   "\"foo\"");
        assert_eq!(Integer(10).to_string(),
                   "10");
        assert_eq!(Float(10.0).to_string(),
                   "10.0");
        assert_eq!(Float(2.4).to_string(),
                   "2.4");
        assert_eq!(Boolean(true).to_string(),
                   "true");
        assert_eq!(Datetime("test".to_string()).to_string(),
                   "test");
        assert_eq!(Array(vec![]).to_string(),
                   "[]");
        assert_eq!(Array(vec![Integer(1), Integer(2)]).to_string(),
                   "[1, 2]");
    }

    #[test]
    fn table() {
        assert_eq!(Table(map! { }).to_string(),
                   "");
        assert_eq!(Table(map! { "test" => Integer(2) }).to_string(),
                   "test = 2\n");
        assert_eq!(Table(map! {
                        "test" => Integer(2),
                        "test2" => Table(map! {
                            "test" => String("wut".to_string())
                        })
                   }).to_string(),
                   "test = 2\n\
                    \n\
                    [test2]\n\
                    test = \"wut\"\n");
        assert_eq!(Table(map! {
                        "test" => Integer(2),
                        "test2" => Table(map! {
                            "test" => String("wut".to_string())
                        })
                   }).to_string(),
                   "test = 2\n\
                    \n\
                    [test2]\n\
                    test = \"wut\"\n");
        assert_eq!(Table(map! {
                        "test" => Integer(2),
                        "test2" => Array(vec![Table(map! {
                            "test" => String("wut".to_string())
                        })])
                   }).to_string(),
                   "test = 2\n\
                    \n\
                    [[test2]]\n\
                    test = \"wut\"\n");
        assert_eq!(Table(map! {
                        "foo.bar" => Integer(2),
                        "foo\"bar" => Integer(2)
                   }).to_string(),
                   "\"foo\\\"bar\" = 2\n\
                    \"foo.bar\" = 2\n");
    }
}
