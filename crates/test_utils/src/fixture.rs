//! Defines `Fixture` -- a convenient way to describe the initial state of
//! rust-analyzer database from a single string.

use rustc_hash::FxHashMap;
use stdx::{lines_with_ends, split_once, trim_indent};

#[derive(Debug, Eq, PartialEq)]
pub struct Fixture {
    pub path: String,
    pub text: String,
    pub krate: Option<String>,
    pub deps: Vec<String>,
    pub cfg_atoms: Vec<String>,
    pub cfg_key_values: Vec<(String, String)>,
    pub edition: Option<String>,
    pub env: FxHashMap<String, String>,
    pub introduce_new_source_root: bool,
}

impl Fixture {
    /// Parses text which looks like this:
    ///
    ///  ```not_rust
    ///  //- some meta
    ///  line 1
    ///  line 2
    ///  // - other meta
    ///  ```
    pub fn parse(ra_fixture: &str) -> Vec<Fixture> {
        let fixture = trim_indent(ra_fixture);

        let mut res: Vec<Fixture> = Vec::new();

        let default = if ra_fixture.contains("//-") { None } else { Some("//- /main.rs") };

        for (ix, line) in default.into_iter().chain(lines_with_ends(&fixture)).enumerate() {
            if line.contains("//-") {
                assert!(
                    line.starts_with("//-"),
                    "Metadata line {} has invalid indentation. \
                     All metadata lines need to have the same indentation.\n\
                     The offending line: {:?}",
                    ix,
                    line
                );
            }

            if line.starts_with("//-") {
                let meta = Fixture::parse_meta_line(line);
                res.push(meta)
            } else if let Some(entry) = res.last_mut() {
                entry.text.push_str(line);
            }
        }

        res
    }

    //- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b env:OUTDIR=path/to,OTHER=foo
    fn parse_meta_line(meta: &str) -> Fixture {
        assert!(meta.starts_with("//-"));
        let meta = meta["//-".len()..].trim();
        let components = meta.split_ascii_whitespace().collect::<Vec<_>>();

        let path = components[0].to_string();
        assert!(path.starts_with('/'));

        let mut krate = None;
        let mut deps = Vec::new();
        let mut edition = None;
        let mut cfg_atoms = Vec::new();
        let mut cfg_key_values = Vec::new();
        let mut env = FxHashMap::default();
        let mut introduce_new_source_root = false;
        for component in components[1..].iter() {
            let (key, value) = split_once(component, ':').unwrap();
            match key {
                "crate" => krate = Some(value.to_string()),
                "deps" => deps = value.split(',').map(|it| it.to_string()).collect(),
                "edition" => edition = Some(value.to_string()),
                "cfg" => {
                    for entry in value.split(',') {
                        match split_once(entry, '=') {
                            Some((k, v)) => cfg_key_values.push((k.to_string(), v.to_string())),
                            None => cfg_atoms.push(entry.to_string()),
                        }
                    }
                }
                "env" => {
                    for key in value.split(',') {
                        if let Some((k, v)) = split_once(key, '=') {
                            env.insert(k.into(), v.into());
                        }
                    }
                }
                "new_source_root" => introduce_new_source_root = true,
                _ => panic!("bad component: {:?}", component),
            }
        }

        Fixture {
            path,
            text: String::new(),
            krate,
            deps,
            cfg_atoms,
            cfg_key_values,
            edition,
            env,
            introduce_new_source_root,
        }
    }
}

#[test]
#[should_panic]
fn parse_fixture_checks_further_indented_metadata() {
    Fixture::parse(
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
fn parse_fixture_gets_full_meta() {
    let parsed = Fixture::parse(
        r"
    //- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b,atom env:OUTDIR=path/to,OTHER=foo
    mod m;
    ",
    );
    assert_eq!(1, parsed.len());

    let meta = &parsed[0];
    assert_eq!("mod m;\n", meta.text);

    assert_eq!("foo", meta.krate.as_ref().unwrap());
    assert_eq!("/lib.rs", meta.path);
    assert_eq!(2, meta.env.len());
}
