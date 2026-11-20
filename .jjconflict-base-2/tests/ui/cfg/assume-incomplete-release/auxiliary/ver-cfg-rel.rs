extern crate proc_macro;
use proc_macro::{TokenStream, TokenTree as Tt};
use std::str::FromStr;

// String containing the current version number of the tip, i.e. "1.41.2"
static VERSION_NUMBER: &str = include_str!("../../../../../src/version");

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Version {
    major: i16,
    minor: i16,
    patch: i16,
}

fn parse_version(s: &str) -> Option<Version> {
    let mut digits = s.splitn(3, '.');
    let major = digits.next()?.parse().ok()?;
    let minor = digits.next()?.parse().ok()?;
    let patch = digits.next().unwrap_or("0").trim().parse().ok()?;
    Some(Version { major, minor, patch })
}

#[proc_macro_attribute]
/// Emits a #[cfg(version)] relative to the current one, so passing
/// -1 as argument on compiler 1.50 will emit #[cfg(version("1.49.0"))],
/// while 1 will emit #[cfg(version("1.51.0"))]
pub fn ver_cfg_rel(attr: TokenStream, input: TokenStream) -> TokenStream {
    let mut v_rel = None;
    for a in attr.into_iter() {
        match a {
            Tt::Literal(l) => {
                let mut s = l.to_string();
                let s = s.trim_matches('"');
                let v: i16 = s.parse().unwrap();
                v_rel = Some(v);
                break;
            },
            _ => panic!("{:?}", a),
        }
    }
    let v_rel = v_rel.unwrap();

    let mut v = parse_version(VERSION_NUMBER).unwrap();
    v.minor += v_rel;

    let attr_str = format!("#[cfg(version(\"{}.{}.{}\"))]", v.major, v.minor, v.patch);
    let mut res = Vec::<Tt>::new();
    res.extend(TokenStream::from_str(&attr_str).unwrap().into_iter());
    res.extend(input.into_iter());
    res.into_iter().collect()
}
