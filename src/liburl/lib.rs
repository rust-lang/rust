// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Types/fns concerning URLs (see RFC 3986)

#![crate_name = "url"]
#![deprecated="This is being removed. Use rust-url instead. http://servo.github.io/rust-url/"]
#![allow(deprecated)]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/master/",
       html_playground_url = "http://play.rust-lang.org/")]
#![feature(default_type_params)]

use std::collections::HashMap;
use std::fmt;
use std::from_str::FromStr;
use std::hash;
use std::uint;
use std::path::BytesContainer;

/// A Uniform Resource Locator (URL).  A URL is a form of URI (Uniform Resource
/// Identifier) that includes network location information, such as hostname or
/// port number.
///
/// # Example
///
/// ```rust
/// # #![allow(deprecated)]
/// use url::Url;
///
/// let raw = "https://username@example.com:8080/foo/bar?baz=qux#quz";
/// match Url::parse(raw) {
///     Ok(u) => println!("Parsed '{}'", u),
///     Err(e) => println!("Couldn't parse '{}': {}", raw, e),
/// }
/// ```
#[deriving(Clone, PartialEq, Eq)]
pub struct Url {
    /// The scheme part of a URL, such as `https` in the above example.
    pub scheme: String,
    /// A URL subcomponent for user authentication.  `username` in the above example.
    pub user: Option<UserInfo>,
    /// A domain name or IP address.  For example, `example.com`.
    pub host: String,
    /// A TCP port number, for example `8080`.
    pub port: Option<u16>,
    /// The path component of a URL, for example `/foo/bar?baz=qux#quz`.
    pub path: Path,
}

#[deriving(Clone, PartialEq, Eq)]
pub struct Path {
    /// The path component of a URL, for example `/foo/bar`.
    pub path: String,
    /// The query component of a URL.
    /// `vec![("baz".to_string(), "qux".to_string())]` represents the fragment
    /// `baz=qux` in the above example.
    pub query: Query,
    /// The fragment component, such as `quz`. Not including the leading `#` character.
    pub fragment: Option<String>
}

/// An optional subcomponent of a URI authority component.
#[deriving(Clone, PartialEq, Eq)]
pub struct UserInfo {
    /// The user name.
    pub user: String,
    /// Password or other scheme-specific authentication information.
    pub pass: Option<String>
}

/// Represents the query component of a URI.
pub type Query = Vec<(String, String)>;

impl Url {
    pub fn new(scheme: String,
               user: Option<UserInfo>,
               host: String,
               port: Option<u16>,
               path: String,
               query: Query,
               fragment: Option<String>)
               -> Url {
        Url {
            scheme: scheme,
            user: user,
            host: host,
            port: port,
            path: Path::new(path, query, fragment)
        }
    }

    /// Parses a URL, converting it from a string to a `Url` representation.
    ///
    /// # Arguments
    /// * rawurl - a string representing the full URL, including scheme.
    ///
    /// # Return value
    ///
    /// `Err(e)` if the string did not represent a valid URL, where `e` is a
    /// `String` error message. Otherwise, `Ok(u)` where `u` is a `Url` struct
    /// representing the URL.
    pub fn parse(rawurl: &str) -> DecodeResult<Url> {
        // scheme
        let (scheme, rest) = try!(get_scheme(rawurl));

        // authority
        let (userinfo, host, port, rest) = try!(get_authority(rest));

        // path
        let has_authority = host.len() > 0;
        let (path, rest) = try!(get_path(rest, has_authority));

        // query and fragment
        let (query, fragment) = try!(get_query_fragment(rest));

        let url = Url::new(scheme.to_string(),
                            userinfo,
                            host.to_string(),
                            port,
                            path,
                            query,
                            fragment);
        Ok(url)
    }
}

#[deprecated="use `Url::parse`"]
pub fn from_str(s: &str) -> Result<Url, String> {
    Url::parse(s)
}

impl Path {
    pub fn new(path: String,
               query: Query,
               fragment: Option<String>)
               -> Path {
        Path {
            path: path,
            query: query,
            fragment: fragment,
        }
    }

    /// Parses a URL path, converting it from a string to a `Path` representation.
    ///
    /// # Arguments
    /// * rawpath - a string representing the path component of a URL.
    ///
    /// # Return value
    ///
    /// `Err(e)` if the string did not represent a valid URL path, where `e` is a
    /// `String` error message. Otherwise, `Ok(p)` where `p` is a `Path` struct
    /// representing the URL path.
    pub fn parse(rawpath: &str) -> DecodeResult<Path> {
        let (path, rest) = try!(get_path(rawpath, false));

        // query and fragment
        let (query, fragment) = try!(get_query_fragment(rest.as_slice()));

        Ok(Path{ path: path, query: query, fragment: fragment })
    }
}

#[deprecated="use `Path::parse`"]
pub fn path_from_str(s: &str) -> Result<Path, String> {
    Path::parse(s)
}

impl UserInfo {
    #[inline]
    pub fn new(user: String, pass: Option<String>) -> UserInfo {
        UserInfo { user: user, pass: pass }
    }
}

fn encode_inner<T: BytesContainer>(c: T, full_url: bool) -> String {
    c.container_as_bytes().iter().fold(String::new(), |mut out, &b| {
        match b as char {
            // unreserved:
            'A' .. 'Z'
            | 'a' .. 'z'
            | '0' .. '9'
            | '-' | '.' | '_' | '~' => out.push_char(b as char),

            // gen-delims:
            ':' | '/' | '?' | '#' | '[' | ']' | '@' |
            // sub-delims:
            '!' | '$' | '&' | '"' | '(' | ')' | '*' |
            '+' | ',' | ';' | '='
                if full_url => out.push_char(b as char),

            ch => out.push_str(format!("%{:02X}", ch as uint).as_slice()),
        };

        out
    })
}

/// Encodes a URI by replacing reserved characters with percent-encoded
/// character sequences.
///
/// This function is compliant with RFC 3986.
///
/// # Example
///
/// ```rust
/// # #![allow(deprecated)]
/// use url::encode;
///
/// let url = encode("https://example.com/Rust (programming language)");
/// println!("{}", url); // https://example.com/Rust%20(programming%20language)
/// ```
pub fn encode<T: BytesContainer>(container: T) -> String {
    encode_inner(container, true)
}


/// Encodes a URI component by replacing reserved characters with percent-
/// encoded character sequences.
///
/// This function is compliant with RFC 3986.
pub fn encode_component<T: BytesContainer>(container: T) -> String {
    encode_inner(container, false)
}

pub type DecodeResult<T> = Result<T, String>;

/// Decodes a percent-encoded string representing a URI.
///
/// This will only decode escape sequences generated by `encode`.
///
/// # Example
///
/// ```rust
/// # #![allow(deprecated)]
/// use url::decode;
///
/// let url = decode("https://example.com/Rust%20(programming%20language)");
/// println!("{}", url); // https://example.com/Rust (programming language)
/// ```
pub fn decode<T: BytesContainer>(container: T) -> DecodeResult<String> {
    decode_inner(container, true)
}

/// Decode a string encoded with percent encoding.
pub fn decode_component<T: BytesContainer>(container: T) -> DecodeResult<String> {
    decode_inner(container, false)
}

fn decode_inner<T: BytesContainer>(c: T, full_url: bool) -> DecodeResult<String> {
    let mut out = String::new();
    let mut iter = c.container_as_bytes().iter().map(|&b| b);

    loop {
        match iter.next() {
            Some(b) => match b as char {
                '%' => {
                    let bytes = match (iter.next(), iter.next()) {
                        (Some(one), Some(two)) => [one as u8, two as u8],
                        _ => return Err(format!("Malformed input: found '%' \
                                                without two trailing bytes")),
                    };

                    // Only decode some characters if full_url:
                    match uint::parse_bytes(bytes, 16u).unwrap() as u8 as char {
                        // gen-delims:
                        ':' | '/' | '?' | '#' | '[' | ']' | '@' |

                        // sub-delims:
                        '!' | '$' | '&' | '"' | '(' | ')' | '*' |
                        '+' | ',' | ';' | '='
                            if full_url => {
                            out.push_char('%');
                            out.push_char(bytes[0u] as char);
                            out.push_char(bytes[1u] as char);
                        }

                        ch => out.push_char(ch)
                    }
                }
                ch => out.push_char(ch)
            },
            None => return Ok(out),
        }
    }
}

/// Encode a hashmap to the 'application/x-www-form-urlencoded' media type.
pub fn encode_form_urlencoded(m: &HashMap<String, Vec<String>>) -> String {
    fn encode_plus<T: Str>(s: &T) -> String {
        s.as_slice().bytes().fold(String::new(), |mut out, b| {
            match b as char {
              'A' .. 'Z'
              | 'a' .. 'z'
              | '0' .. '9'
              | '_' | '.' | '-' => out.push_char(b as char),
              ' ' => out.push_char('+'),
              ch => out.push_str(format!("%{:X}", ch as uint).as_slice())
            }

            out
        })
    }

    let mut first = true;
    m.iter().fold(String::new(), |mut out, (key, values)| {
        let key = encode_plus(key);

        for value in values.iter() {
            if first {
                first = false;
            } else {
                out.push_char('&');
            }

            out.push_str(key.as_slice());
            out.push_char('=');
            out.push_str(encode_plus(value).as_slice());
        }

        out
    })
}

/// Decode a string encoded with the 'application/x-www-form-urlencoded' media
/// type into a hashmap.
pub fn decode_form_urlencoded(s: &[u8])
                            -> DecodeResult<HashMap<String, Vec<String>>> {
    fn maybe_push_value(map: &mut HashMap<String, Vec<String>>,
                        key: String,
                        value: String) {
        if key.len() > 0 && value.len() > 0 {
            let values = map.find_or_insert_with(key, |_| vec!());
            values.push(value);
        }
    }

    let mut out = HashMap::new();
    let mut iter = s.iter().map(|&x| x);

    let mut key = String::new();
    let mut value = String::new();
    let mut parsing_key = true;

    loop {
        match iter.next() {
            Some(b) => match b as char {
                '&' | ';' => {
                    maybe_push_value(&mut out, key, value);

                    parsing_key = true;
                    key = String::new();
                    value = String::new();
                }
                '=' => parsing_key = false,
                ch => {
                    let ch = match ch {
                        '%' => {
                            let bytes = match (iter.next(), iter.next()) {
                                (Some(one), Some(two)) => [one as u8, two as u8],
                                _ => return Err(format!("Malformed input: found \
                                                '%' without two trailing bytes"))
                            };

                            uint::parse_bytes(bytes, 16u).unwrap() as u8 as char
                        }
                        '+' => ' ',
                        ch => ch
                    };

                    if parsing_key {
                        key.push_char(ch)
                    } else {
                        value.push_char(ch)
                    }
                }
            },
            None => {
                maybe_push_value(&mut out, key, value);
                return Ok(out)
            }
        }
    }
}

fn split_char_first(s: &str, c: char) -> (&str, &str) {
    let mut iter = s.splitn(1, c);

    match (iter.next(), iter.next()) {
        (Some(a), Some(b)) => (a, b),
        (Some(a), None) => (a, ""),
        (None, _) => unreachable!(),
    }
}

impl fmt::Show for UserInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.pass {
            Some(ref pass) => write!(f, "{}:{}@", self.user, *pass),
            None => write!(f, "{}@", self.user),
        }
    }
}

fn query_from_str(rawquery: &str) -> DecodeResult<Query> {
    let mut query: Query = vec!();
    if !rawquery.is_empty() {
        for p in rawquery.split('&') {
            let (k, v) = split_char_first(p, '=');
            query.push((try!(decode_component(k)),
                        try!(decode_component(v))));
        }
    }

    Ok(query)
}

/// Converts an instance of a URI `Query` type to a string.
///
/// # Example
///
/// ```rust
/// # #![allow(deprecated)]
/// let query = vec![("title".to_string(), "The Village".to_string()),
///                  ("north".to_string(), "52.91".to_string()),
///                  ("west".to_string(), "4.10".to_string())];
/// println!("{}", url::query_to_str(&query));  // title=The%20Village&north=52.91&west=4.10
/// ```
pub fn query_to_str(query: &Query) -> String {
    query.iter().enumerate().fold(String::new(), |mut out, (i, &(ref k, ref v))| {
        if i != 0 {
            out.push_char('&');
        }

        out.push_str(encode_component(k.as_slice()).as_slice());
        out.push_char('=');
        out.push_str(encode_component(v.as_slice()).as_slice());
        out
    })
}

/// Returns a tuple of the URI scheme and the rest of the URI, or a parsing error.
///
/// Does not include the separating `:` character.
///
/// # Example
///
/// ```rust
/// # #![allow(deprecated)]
/// use url::get_scheme;
///
/// let scheme = match get_scheme("https://example.com/") {
///     Ok((sch, _)) => sch,
///     Err(_) => "(None)",
/// };
/// println!("Scheme in use: {}.", scheme); // Scheme in use: https.
/// ```
pub fn get_scheme(rawurl: &str) -> DecodeResult<(&str, &str)> {
    for (i,c) in rawurl.chars().enumerate() {
        let result = match c {
            'A' .. 'Z'
            | 'a' .. 'z' => continue,
            '0' .. '9' | '+' | '-' | '.' => {
                if i != 0 { continue }

                Err("url: Scheme must begin with a letter.".to_string())
            }
            ':' => {
                if i == 0 {
                    Err("url: Scheme cannot be empty.".to_string())
                } else {
                    Ok((rawurl.slice(0,i), rawurl.slice(i+1,rawurl.len())))
                }
            }
            _ => Err("url: Invalid character in scheme.".to_string()),
        };

        return result;
    }

    Err("url: Scheme must be terminated with a colon.".to_string())
}

// returns userinfo, host, port, and unparsed part, or an error
fn get_authority(rawurl: &str) ->
    DecodeResult<(Option<UserInfo>, &str, Option<u16>, &str)> {
    enum State {
        Start, // starting state
        PassHostPort, // could be in user or port
        Ip6Port, // either in ipv6 host or port
        Ip6Host, // are in an ipv6 host
        InHost, // are in a host - may be ipv6, but don't know yet
        InPort // are in port
    }

    #[deriving(Clone, PartialEq)]
    enum Input {
        Digit, // all digits
        Hex, // digits and letters a-f
        Unreserved // all other legal characters
    }

    if !rawurl.starts_with("//") {
        // there is no authority.
        return Ok((None, "", None, rawurl));
    }

    let len = rawurl.len();
    let mut st = Start;
    let mut input = Digit; // most restricted, start here.

    let mut userinfo = None;
    let mut host = "";
    let mut port = None;

    let mut colon_count = 0u;
    let mut pos = 0;
    let mut begin = 2;
    let mut end = len;

    for (i,c) in rawurl.chars().enumerate()
                               // ignore the leading '//' handled by early return
                               .skip(2) {
        // deal with input class first
        match c {
            '0' .. '9' => (),
            'A' .. 'F'
            | 'a' .. 'f' => {
                if input == Digit {
                    input = Hex;
                }
            }
            'G' .. 'Z'
            | 'g' .. 'z'
            | '-' | '.' | '_' | '~' | '%'
            | '&' |'\'' | '(' | ')' | '+'
            | '!' | '*' | ',' | ';' | '=' => input = Unreserved,
            ':' | '@' | '?' | '#' | '/' => {
                // separators, don't change anything
            }
            _ => return Err("Illegal character in authority".to_string()),
        }

        // now process states
        match c {
          ':' => {
            colon_count += 1;
            match st {
              Start => {
                pos = i;
                st = PassHostPort;
              }
              PassHostPort => {
                // multiple colons means ipv6 address.
                if input == Unreserved {
                    return Err(
                        "Illegal characters in IPv6 address.".to_string());
                }
                st = Ip6Host;
              }
              InHost => {
                pos = i;
                if input == Unreserved {
                    // must be port
                    host = rawurl.slice(begin, i);
                    st = InPort;
                } else {
                    // can't be sure whether this is an ipv6 address or a port
                    st = Ip6Port;
                }
              }
              Ip6Port => {
                if input == Unreserved {
                    return Err("Illegal characters in authority.".to_string());
                }
                st = Ip6Host;
              }
              Ip6Host => {
                if colon_count > 7 {
                    host = rawurl.slice(begin, i);
                    pos = i;
                    st = InPort;
                }
              }
              _ => return Err("Invalid ':' in authority.".to_string()),
            }
            input = Digit; // reset input class
          }

          '@' => {
            input = Digit; // reset input class
            colon_count = 0; // reset count
            match st {
              Start => {
                let user = rawurl.slice(begin, i).to_string();
                userinfo = Some(UserInfo::new(user, None));
                st = InHost;
              }
              PassHostPort => {
                let user = rawurl.slice(begin, pos).to_string();
                let pass = rawurl.slice(pos+1, i).to_string();
                userinfo = Some(UserInfo::new(user, Some(pass)));
                st = InHost;
              }
              _ => return Err("Invalid '@' in authority.".to_string()),
            }
            begin = i+1;
          }

          '?' | '#' | '/' => {
            end = i;
            break;
          }
          _ => ()
        }
    }

    // finish up
    match st {
      Start => host = rawurl.slice(begin, end),
      PassHostPort
      | Ip6Port => {
        if input != Digit {
            return Err("Non-digit characters in port.".to_string());
        }
        host = rawurl.slice(begin, pos);
        port = Some(rawurl.slice(pos+1, end));
      }
      Ip6Host
      | InHost => host = rawurl.slice(begin, end),
      InPort => {
        if input != Digit {
            return Err("Non-digit characters in port.".to_string());
        }
        port = Some(rawurl.slice(pos+1, end));
      }
    }

    let rest = rawurl.slice(end, len);
    // If we have a port string, ensure it parses to u16.
    let port = match port {
        None => None,
        opt => match opt.and_then(|p| FromStr::from_str(p)) {
            None => return Err(format!("Failed to parse port: {}", port)),
            opt => opt
        }
    };

    Ok((userinfo, host, port, rest))
}


// returns the path and unparsed part of url, or an error
fn get_path(rawurl: &str, is_authority: bool) -> DecodeResult<(String, &str)> {
    let len = rawurl.len();
    let mut end = len;
    for (i,c) in rawurl.chars().enumerate() {
        match c {
          'A' .. 'Z'
          | 'a' .. 'z'
          | '0' .. '9'
          | '&' |'\'' | '(' | ')' | '.'
          | '@' | ':' | '%' | '/' | '+'
          | '!' | '*' | ',' | ';' | '='
          | '_' | '-' | '~' => continue,
          '?' | '#' => {
            end = i;
            break;
          }
          _ => return Err("Invalid character in path.".to_string())
        }
    }

    if is_authority && end != 0 && !rawurl.starts_with("/") {
        Err("Non-empty path must begin with \
            '/' in presence of authority.".to_string())
    } else {
        Ok((try!(decode_component(rawurl.slice(0, end))),
            rawurl.slice(end, len)))
    }
}

// returns the parsed query and the fragment, if present
fn get_query_fragment(rawurl: &str) -> DecodeResult<(Query, Option<String>)> {
    let (before_fragment, raw_fragment) = split_char_first(rawurl, '#');

    // Parse the fragment if available
    let fragment = match raw_fragment {
        "" => None,
        raw => Some(try!(decode_component(raw)))
    };

    match before_fragment.slice_shift_char() {
        (Some('?'), rest) => Ok((try!(query_from_str(rest)), fragment)),
        (None, "") => Ok((vec!(), fragment)),
        _ => Err(format!("Query didn't start with '?': '{}..'", before_fragment)),
    }
}

impl FromStr for Url {
    fn from_str(s: &str) -> Option<Url> {
        Url::parse(s).ok()
    }
}

impl FromStr for Path {
    fn from_str(s: &str) -> Option<Path> {
        Path::parse(s).ok()
    }
}

impl fmt::Show for Url {
    /// Converts a URL from `Url` to string representation.
    ///
    /// # Returns
    ///
    /// A string that contains the formatted URL. Note that this will usually
    /// be an inverse of `from_str` but might strip out unneeded separators;
    /// for example, "http://somehost.com?", when parsed and formatted, will
    /// result in just "http://somehost.com".
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{}:", self.scheme));

        if !self.host.is_empty() {
            try!(write!(f, "//"));
            match self.user {
                Some(ref user) => try!(write!(f, "{}", *user)),
                None => {}
            }
            match self.port {
                Some(ref port) => try!(write!(f, "{}:{}", self.host,
                                                *port)),
                None => try!(write!(f, "{}", self.host)),
            }
        }

        write!(f, "{}", self.path)
    }
}

impl fmt::Show for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{}", self.path));
        if !self.query.is_empty() {
            try!(write!(f, "?{}", query_to_str(&self.query)))
        }

        match self.fragment {
            Some(ref fragment) => {
                write!(f, "#{}", encode_component(fragment.as_slice()))
            }
            None => Ok(())
        }
    }
}

impl<S: hash::Writer> hash::Hash<S> for Url {
    fn hash(&self, state: &mut S) {
        self.to_string().hash(state)
    }
}

impl<S: hash::Writer> hash::Hash<S> for Path {
    fn hash(&self, state: &mut S) {
        self.to_string().hash(state)
    }
}

// Put a few tests outside of the 'test' module so they can test the internal
// functions and those functions don't need 'pub'

#[test]
fn test_split_char_first() {
    let (u,v) = split_char_first("hello, sweet world", ',');
    assert_eq!(u, "hello");
    assert_eq!(v, " sweet world");

    let (u,v) = split_char_first("hello sweet world", ',');
    assert_eq!(u, "hello sweet world");
    assert_eq!(v, "");
}

#[test]
fn test_get_authority() {
    let (u, h, p, r) = get_authority(
        "//user:pass@rust-lang.org/something").unwrap();
    assert_eq!(u, Some(UserInfo::new("user".to_string(), Some("pass".to_string()))));
    assert_eq!(h, "rust-lang.org");
    assert!(p.is_none());
    assert_eq!(r, "/something");

    let (u, h, p, r) = get_authority(
        "//rust-lang.org:8000?something").unwrap();
    assert!(u.is_none());
    assert_eq!(h, "rust-lang.org");
    assert_eq!(p, Some(8000));
    assert_eq!(r, "?something");

    let (u, h, p, r) = get_authority("//rust-lang.org#blah").unwrap();
    assert!(u.is_none());
    assert_eq!(h, "rust-lang.org");
    assert!(p.is_none());
    assert_eq!(r, "#blah");

    // ipv6 tests
    let (_, h, _, _) = get_authority(
        "//2001:0db8:85a3:0042:0000:8a2e:0370:7334#blah").unwrap();
    assert_eq!(h, "2001:0db8:85a3:0042:0000:8a2e:0370:7334");

    let (_, h, p, _) = get_authority(
        "//2001:0db8:85a3:0042:0000:8a2e:0370:7334:8000#blah").unwrap();
    assert_eq!(h, "2001:0db8:85a3:0042:0000:8a2e:0370:7334");
    assert_eq!(p, Some(8000));

    let (u, h, p, _) = get_authority(
        "//us:p@2001:0db8:85a3:0042:0000:8a2e:0370:7334:8000#blah"
    ).unwrap();
    assert_eq!(u, Some(UserInfo::new("us".to_string(), Some("p".to_string()))));
    assert_eq!(h, "2001:0db8:85a3:0042:0000:8a2e:0370:7334");
    assert_eq!(p, Some(8000));

    // invalid authorities;
    assert!(get_authority("//user:pass@rust-lang:something").is_err());
    assert!(get_authority("//user@rust-lang:something:/path").is_err());
    assert!(get_authority(
        "//2001:0db8:85a3:0042:0000:8a2e:0370:7334:800a").is_err());
    assert!(get_authority(
        "//2001:0db8:85a3:0042:0000:8a2e:0370:7334:8000:00").is_err());
    // outside u16 range
    assert!(get_authority("//user:pass@rust-lang:65536").is_err());

    // these parse as empty, because they don't start with '//'
    let (_, h, _, _) = get_authority("user:pass@rust-lang").unwrap();
    assert_eq!(h, "");
    let (_, h, _, _) = get_authority("rust-lang.org").unwrap();
    assert_eq!(h, "");
}

#[test]
fn test_get_path() {
    let (p, r) = get_path("/something+%20orother", true).unwrap();
    assert_eq!(p, "/something+ orother".to_string());
    assert_eq!(r, "");
    let (p, r) = get_path("test@email.com#fragment", false).unwrap();
    assert_eq!(p, "test@email.com".to_string());
    assert_eq!(r, "#fragment");
    let (p, r) = get_path("/gen/:addr=?q=v", false).unwrap();
    assert_eq!(p, "/gen/:addr=".to_string());
    assert_eq!(r, "?q=v");

    //failure cases
    assert!(get_path("something?q", true).is_err());
}

#[cfg(test)]
mod tests {
    use {encode_form_urlencoded, decode_form_urlencoded, decode, encode,
        encode_component, decode_component, UserInfo, get_scheme, Url, Path};

    use std::collections::HashMap;
    use std::path::BytesContainer;

    #[test]
    fn test_url_parse() {
        let url = "http://user:pass@rust-lang.org:8080/doc/~u?s=v#something";
        let u = from_str::<Url>(url).unwrap();

        assert_eq!(u.scheme, "http".to_string());
        assert_eq!(u.user, Some(UserInfo::new("user".to_string(), Some("pass".to_string()))));
        assert_eq!(u.host, "rust-lang.org".to_string());
        assert_eq!(u.port, Some(8080));
        assert_eq!(u.path.path, "/doc/~u".to_string());
        assert_eq!(u.path.query, vec!(("s".to_string(), "v".to_string())));
        assert_eq!(u.path.fragment, Some("something".to_string()));
    }

    #[test]
    fn test_path_parse() {
        let path = "/doc/~u?s=v#something";
        let u = from_str::<Path>(path).unwrap();

        assert_eq!(u.path, "/doc/~u".to_string());
        assert_eq!(u.query, vec!(("s".to_string(), "v".to_string())));
        assert_eq!(u.fragment, Some("something".to_string()));
    }

    #[test]
    fn test_url_parse_host_slash() {
        let urlstr = "http://0.42.42.42/";
        let url = from_str::<Url>(urlstr).unwrap();
        assert_eq!(url.host, "0.42.42.42".to_string());
        assert_eq!(url.path.path, "/".to_string());
    }

    #[test]
    fn test_path_parse_host_slash() {
        let pathstr = "/";
        let path = from_str::<Path>(pathstr).unwrap();
        assert_eq!(path.path, "/".to_string());
    }

    #[test]
    fn test_url_host_with_port() {
        let urlstr = "scheme://host:1234";
        let url = from_str::<Url>(urlstr).unwrap();
        assert_eq!(url.scheme, "scheme".to_string());
        assert_eq!(url.host, "host".to_string());
        assert_eq!(url.port, Some(1234));
        // is empty path really correct? Other tests think so
        assert_eq!(url.path.path, "".to_string());

        let urlstr = "scheme://host:1234/";
        let url = from_str::<Url>(urlstr).unwrap();
        assert_eq!(url.scheme, "scheme".to_string());
        assert_eq!(url.host, "host".to_string());
        assert_eq!(url.port, Some(1234));
        assert_eq!(url.path.path, "/".to_string());
    }

    #[test]
    fn test_url_with_underscores() {
        let urlstr = "http://dotcom.com/file_name.html";
        let url = from_str::<Url>(urlstr).unwrap();
        assert_eq!(url.path.path, "/file_name.html".to_string());
    }

    #[test]
    fn test_path_with_underscores() {
        let pathstr = "/file_name.html";
        let path = from_str::<Path>(pathstr).unwrap();
        assert_eq!(path.path, "/file_name.html".to_string());
    }

    #[test]
    fn test_url_with_dashes() {
        let urlstr = "http://dotcom.com/file-name.html";
        let url = from_str::<Url>(urlstr).unwrap();
        assert_eq!(url.path.path, "/file-name.html".to_string());
    }

    #[test]
    fn test_path_with_dashes() {
        let pathstr = "/file-name.html";
        let path = from_str::<Path>(pathstr).unwrap();
        assert_eq!(path.path, "/file-name.html".to_string());
    }

    #[test]
    fn test_no_scheme() {
        assert!(get_scheme("noschemehere.html").is_err());
    }

    #[test]
    fn test_invalid_scheme_errors() {
        assert!(Url::parse("99://something").is_err());
        assert!(Url::parse("://something").is_err());
    }

    #[test]
    fn test_full_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org/doc?s=v#something";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_userless_url_parse_and_format() {
        let url = "http://rust-lang.org/doc?s=v#something";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_queryless_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org/doc#something";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_empty_query_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org/doc?#something";
        let should_be = "http://user:pass@rust-lang.org/doc#something";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), should_be);
    }

    #[test]
    fn test_fragmentless_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org/doc?q=v";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_minimal_url_parse_and_format() {
        let url = "http://rust-lang.org/doc";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_url_with_port_parse_and_format() {
        let url = "http://rust-lang.org:80/doc";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_scheme_host_only_url_parse_and_format() {
        let url = "http://rust-lang.org";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_pathless_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org?q=v#something";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_scheme_host_fragment_only_url_parse_and_format() {
        let url = "http://rust-lang.org#something";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_url_component_encoding() {
        let url = "http://rust-lang.org/doc%20uments?ba%25d%20=%23%26%2B";
        let u = from_str::<Url>(url).unwrap();
        assert!(u.path.path == "/doc uments".to_string());
        assert!(u.path.query == vec!(("ba%d ".to_string(), "#&+".to_string())));
    }

    #[test]
    fn test_path_component_encoding() {
        let path = "/doc%20uments?ba%25d%20=%23%26%2B";
        let p = from_str::<Path>(path).unwrap();
        assert!(p.path == "/doc uments".to_string());
        assert!(p.query == vec!(("ba%d ".to_string(), "#&+".to_string())));
    }

    #[test]
    fn test_url_without_authority() {
        let url = "mailto:test@email.com";
        let u = from_str::<Url>(url).unwrap();
        assert_eq!(format!("{}", u).as_slice(), url);
    }

    #[test]
    fn test_encode() {
        fn t<T: BytesContainer>(input: T, expected: &str) {
            assert_eq!(encode(input), expected.to_string())
        }

        t("", "");
        t("http://example.com", "http://example.com");
        t("foo bar% baz", "foo%20bar%25%20baz");
        t(" ", "%20");
        t("!", "!");
        t("\"", "\"");
        t("#", "#");
        t("$", "$");
        t("%", "%25");
        t("&", "&");
        t("'", "%27");
        t("(", "(");
        t(")", ")");
        t("*", "*");
        t("+", "+");
        t(",", ",");
        t("/", "/");
        t(":", ":");
        t(";", ";");
        t("=", "=");
        t("?", "?");
        t("@", "@");
        t("[", "[");
        t("]", "]");
        t("\0", "%00");
        t("\n", "%0A");

        let a: &[_] = &[0u8, 10, 37];
        t(a, "%00%0A%25");
    }

    #[test]
    fn test_encode_component() {
        fn t<T: BytesContainer>(input: T, expected: &str) {
            assert_eq!(encode_component(input), expected.to_string())
        }

        t("", "");
        t("http://example.com", "http%3A%2F%2Fexample.com");
        t("foo bar% baz", "foo%20bar%25%20baz");
        t(" ", "%20");
        t("!", "%21");
        t("#", "%23");
        t("$", "%24");
        t("%", "%25");
        t("&", "%26");
        t("'", "%27");
        t("(", "%28");
        t(")", "%29");
        t("*", "%2A");
        t("+", "%2B");
        t(",", "%2C");
        t("/", "%2F");
        t(":", "%3A");
        t(";", "%3B");
        t("=", "%3D");
        t("?", "%3F");
        t("@", "%40");
        t("[", "%5B");
        t("]", "%5D");
        t("\0", "%00");
        t("\n", "%0A");

        let a: &[_] = &[0u8, 10, 37];
        t(a, "%00%0A%25");
    }

    #[test]
    fn test_decode() {
        fn t<T: BytesContainer>(input: T, expected: &str) {
            assert_eq!(decode(input), Ok(expected.to_string()))
        }

        assert!(decode("sadsadsda%").is_err());
        assert!(decode("waeasd%4").is_err());
        t("", "");
        t("abc/def 123", "abc/def 123");
        t("abc%2Fdef%20123", "abc%2Fdef 123");
        t("%20", " ");
        t("%21", "%21");
        t("%22", "%22");
        t("%23", "%23");
        t("%24", "%24");
        t("%25", "%");
        t("%26", "%26");
        t("%27", "'");
        t("%28", "%28");
        t("%29", "%29");
        t("%2A", "%2A");
        t("%2B", "%2B");
        t("%2C", "%2C");
        t("%2F", "%2F");
        t("%3A", "%3A");
        t("%3B", "%3B");
        t("%3D", "%3D");
        t("%3F", "%3F");
        t("%40", "%40");
        t("%5B", "%5B");
        t("%5D", "%5D");

        t("%00%0A%25".as_bytes(), "\0\n%");
    }

    #[test]
    fn test_decode_component() {
        fn t<T: BytesContainer>(input: T, expected: &str) {
            assert_eq!(decode_component(input), Ok(expected.to_string()))
        }

        assert!(decode_component("asacsa%").is_err());
        assert!(decode_component("acsas%4").is_err());
        t("", "");
        t("abc/def 123", "abc/def 123");
        t("abc%2Fdef%20123", "abc/def 123");
        t("%20", " ");
        t("%21", "!");
        t("%22", "\"");
        t("%23", "#");
        t("%24", "$");
        t("%25", "%");
        t("%26", "&");
        t("%27", "'");
        t("%28", "(");
        t("%29", ")");
        t("%2A", "*");
        t("%2B", "+");
        t("%2C", ",");
        t("%2F", "/");
        t("%3A", ":");
        t("%3B", ";");
        t("%3D", "=");
        t("%3F", "?");
        t("%40", "@");
        t("%5B", "[");
        t("%5D", "]");

        t("%00%0A%25".as_bytes(), "\0\n%");
    }

    #[test]
    fn test_encode_form_urlencoded() {
        let mut m = HashMap::new();
        assert_eq!(encode_form_urlencoded(&m), "".to_string());

        m.insert("".to_string(), vec!());
        m.insert("foo".to_string(), vec!());
        assert_eq!(encode_form_urlencoded(&m), "".to_string());

        let mut m = HashMap::new();
        m.insert("foo".to_string(), vec!("bar".to_string(), "123".to_string()));
        assert_eq!(encode_form_urlencoded(&m), "foo=bar&foo=123".to_string());

        let mut m = HashMap::new();
        m.insert("foo bar".to_string(), vec!("abc".to_string(), "12 = 34".to_string()));
        assert_eq!(encode_form_urlencoded(&m),
                    "foo+bar=abc&foo+bar=12+%3D+34".to_string());
    }

    #[test]
    fn test_decode_form_urlencoded() {
        assert_eq!(decode_form_urlencoded([]).unwrap().len(), 0);

        let s = "a=1&foo+bar=abc&foo+bar=12+%3D+34".as_bytes();
        let form = decode_form_urlencoded(s).unwrap();
        assert_eq!(form.len(), 2);
        assert_eq!(form.get(&"a".to_string()), &vec!("1".to_string()));
        assert_eq!(form.get(&"foo bar".to_string()),
                   &vec!("abc".to_string(), "12 = 34".to_string()));
    }
}
