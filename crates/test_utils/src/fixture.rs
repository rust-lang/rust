use ra_cfg::CfgOptions;
use rustc_hash::FxHashMap;
use stdx::split1;

#[derive(Debug, Eq, PartialEq)]
pub struct FixtureEntry {
    pub meta: FixtureMeta,
    pub text: String,
}

#[derive(Debug, Eq, PartialEq)]
pub enum FixtureMeta {
    Root { path: String },
    File(FileMeta),
}

#[derive(Debug, Eq, PartialEq)]
pub struct FileMeta {
    pub path: String,
    pub crate_name: Option<String>,
    pub deps: Vec<String>,
    pub cfg: CfgOptions,
    pub edition: Option<String>,
    pub env: FxHashMap<String, String>,
}

impl FixtureMeta {
    pub fn path(&self) -> &str {
        match self {
            FixtureMeta::Root { path } => &path,
            FixtureMeta::File(f) => &f.path,
        }
    }

    pub fn crate_name(&self) -> Option<&String> {
        match self {
            FixtureMeta::File(f) => f.crate_name.as_ref(),
            _ => None,
        }
    }

    pub fn cfg_options(&self) -> Option<&CfgOptions> {
        match self {
            FixtureMeta::File(f) => Some(&f.cfg),
            _ => None,
        }
    }

    pub fn edition(&self) -> Option<&String> {
        match self {
            FixtureMeta::File(f) => f.edition.as_ref(),
            _ => None,
        }
    }

    pub fn env(&self) -> impl Iterator<Item = (&String, &String)> {
        struct EnvIter<'a> {
            iter: Option<std::collections::hash_map::Iter<'a, String, String>>,
        }

        impl<'a> EnvIter<'a> {
            fn new(meta: &'a FixtureMeta) -> Self {
                Self {
                    iter: match meta {
                        FixtureMeta::File(f) => Some(f.env.iter()),
                        _ => None,
                    },
                }
            }
        }

        impl<'a> Iterator for EnvIter<'a> {
            type Item = (&'a String, &'a String);
            fn next(&mut self) -> Option<Self::Item> {
                self.iter.as_mut().and_then(|i| i.next())
            }
        }

        EnvIter::new(self)
    }
}

/// Same as `parse_fixture`, except it allow empty fixture
pub fn parse_single_fixture(ra_fixture: &str) -> Option<FixtureEntry> {
    if !ra_fixture.lines().any(|it| it.trim_start().starts_with("//-")) {
        return None;
    }

    let fixtures = parse_fixture(ra_fixture);
    if fixtures.len() > 1 {
        panic!("too many fixtures");
    }
    fixtures.into_iter().nth(0)
}

/// Parses text which looks like this:
///
///  ```not_rust
///  //- some meta
///  line 1
///  line 2
///  // - other meta
///  ```
pub fn parse_fixture(ra_fixture: &str) -> Vec<FixtureEntry> {
    let fixture = indent_first_line(ra_fixture);
    let margin = fixture_margin(&fixture);

    let mut lines = fixture
        .split('\n') // don't use `.lines` to not drop `\r\n`
        .enumerate()
        .filter_map(|(ix, line)| {
            if line.len() >= margin {
                assert!(line[..margin].trim().is_empty());
                let line_content = &line[margin..];
                if !line_content.starts_with("//-") {
                    assert!(
                        !line_content.contains("//-"),
                        r#"Metadata line {} has invalid indentation. All metadata lines need to have the same indentation.
The offending line: {:?}"#,
                        ix,
                        line
                    );
                }
                Some(line_content)
            } else {
                assert!(line.trim().is_empty());
                None
            }
        });

    let mut res: Vec<FixtureEntry> = Vec::new();
    for line in lines.by_ref() {
        if line.starts_with("//-") {
            let meta = line["//-".len()..].trim().to_string();
            let meta = parse_meta(&meta);
            res.push(FixtureEntry { meta, text: String::new() })
        } else if let Some(entry) = res.last_mut() {
            entry.text.push_str(line);
            entry.text.push('\n');
        }
    }
    res
}

//- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b env:OUTDIR=path/to,OTHER=foo
fn parse_meta(meta: &str) -> FixtureMeta {
    let components = meta.split_ascii_whitespace().collect::<Vec<_>>();

    if components[0] == "root" {
        let path = components[1].to_string();
        assert!(path.starts_with("/") && path.ends_with("/"));
        return FixtureMeta::Root { path };
    }

    let path = components[0].to_string();
    assert!(path.starts_with("/"));

    let mut krate = None;
    let mut deps = Vec::new();
    let mut edition = None;
    let mut cfg = CfgOptions::default();
    let mut env = FxHashMap::default();
    for component in components[1..].iter() {
        let (key, value) = split1(component, ':').unwrap();
        match key {
            "crate" => krate = Some(value.to_string()),
            "deps" => deps = value.split(',').map(|it| it.to_string()).collect(),
            "edition" => edition = Some(value.to_string()),
            "cfg" => {
                for key in value.split(',') {
                    match split1(key, '=') {
                        None => cfg.insert_atom(key.into()),
                        Some((k, v)) => cfg.insert_key_value(k.into(), v.into()),
                    }
                }
            }
            "env" => {
                for key in value.split(',') {
                    if let Some((k, v)) = split1(key, '=') {
                        env.insert(k.into(), v.into());
                    }
                }
            }
            _ => panic!("bad component: {:?}", component),
        }
    }

    FixtureMeta::File(FileMeta { path, crate_name: krate, deps, edition, cfg, env })
}

/// Adjusts the indentation of the first line to the minimum indentation of the rest of the lines.
/// This allows fixtures to start off in a different indentation, e.g. to align the first line with
/// the other lines visually:
/// ```
/// let fixture = "//- /lib.rs
///                mod foo;
///                //- /foo.rs
///                fn bar() {}
/// ";
/// assert_eq!(fixture_margin(fixture),
/// "               //- /lib.rs
///                mod foo;
///                //- /foo.rs
///                fn bar() {}
/// ")
/// ```
fn indent_first_line(fixture: &str) -> String {
    if fixture.is_empty() {
        return String::new();
    }
    let mut lines = fixture.lines();
    let first_line = lines.next().unwrap();
    if first_line.contains("//-") {
        let rest = lines.collect::<Vec<_>>().join("\n");
        let fixed_margin = fixture_margin(&rest);
        let fixed_indent = fixed_margin - indent_len(first_line);
        format!("\n{}{}\n{}", " ".repeat(fixed_indent), first_line, rest)
    } else {
        fixture.to_owned()
    }
}

fn fixture_margin(fixture: &str) -> usize {
    fixture
        .lines()
        .filter(|it| it.trim_start().starts_with("//-"))
        .map(indent_len)
        .next()
        .expect("empty fixture")
}

fn indent_len(s: &str) -> usize {
    s.len() - s.trim_start().len()
}

#[test]
#[should_panic]
fn parse_fixture_checks_further_indented_metadata() {
    parse_fixture(
        r"
        //- /lib.rs
          mod bar;

          fn foo() {}
          //- /bar.rs
          pub fn baz() {}
          ",
    );
}

#[test]
fn parse_fixture_can_handle_dedented_first_line() {
    let fixture = "//- /lib.rs
                   mod foo;
                   //- /foo.rs
                   struct Bar;
";
    assert_eq!(
        parse_fixture(fixture),
        parse_fixture(
            "//- /lib.rs
mod foo;
//- /foo.rs
struct Bar;
"
        )
    )
}

#[test]
fn parse_fixture_gets_full_meta() {
    let parsed = parse_fixture(
        r"
    //- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b,atom env:OUTDIR=path/to,OTHER=foo
    mod m;
    ",
    );
    assert_eq!(1, parsed.len());

    let parsed = &parsed[0];
    assert_eq!("mod m;\n\n", parsed.text);

    let meta = &parsed.meta;
    assert_eq!("foo", meta.crate_name().unwrap());
    assert_eq!("/lib.rs", meta.path());
    assert!(meta.cfg_options().is_some());
    assert_eq!(2, meta.env().count());
}
