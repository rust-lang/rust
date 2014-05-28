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

#![crate_id = "url#0.11.0-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/")]
#![feature(default_type_params)]

extern crate collections;

use collections::HashMap;
use std::cmp::Eq;
use std::fmt;
use std::from_str::FromStr;
use std::hash::Hash;
use std::io::BufReader;
use std::string::String;
use std::uint;

/// A Uniform Resource Locator (URL).  A URL is a form of URI (Uniform Resource
/// Identifier) that includes network location information, such as hostname or
/// port number.
///
/// # Example
///
/// ```rust
/// use url::{Url, UserInfo};
///
/// let url = Url { scheme: "https".to_string(),
///                 user: Some(UserInfo { user: "username".to_string(), pass: None }),
///                 host: "example.com".to_string(),
///                 port: Some("8080".to_string()),
///                 path: "/foo/bar".to_string(),
///                 query: vec!(("baz".to_string(), "qux".to_string())),
///                 fragment: Some("quz".to_string()) };
/// // https://username@example.com:8080/foo/bar?baz=qux#quz
/// ```
#[deriving(Clone, Eq, TotalEq)]
pub struct Url {
    /// The scheme part of a URL, such as `https` in the above example.
    pub scheme: String,
    /// A URL subcomponent for user authentication.  `username` in the above example.
    pub user: Option<UserInfo>,
    /// A domain name or IP address.  For example, `example.com`.
    pub host: String,
    /// A TCP port number, for example `8080`.
    pub port: Option<String>,
    /// The path component of a URL, for example `/foo/bar`.
    pub path: String,
    /// The query component of a URL.
    /// `vec!(("baz".to_string(), "qux".to_string()))` represents the fragment
    /// `baz=qux` in the above example.
    pub query: Query,
    /// The fragment component, such as `quz`.  Doesn't include the leading `#` character.
    pub fragment: Option<String>
}

#[deriving(Clone, Eq)]
pub struct Path {
    /// The path component of a URL, for example `/foo/bar`.
    pub path: String,
    /// The query component of a URL.
    /// `vec!(("baz".to_string(), "qux".to_string()))` represents the fragment
    /// `baz=qux` in the above example.
    pub query: Query,
    /// The fragment component, such as `quz`.  Doesn't include the leading `#` character.
    pub fragment: Option<String>
}

/// An optional subcomponent of a URI authority component.
#[deriving(Clone, Eq, TotalEq)]
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
               port: Option<String>,
               path: String,
               query: Query,
               fragment: Option<String>)
               -> Url {
        Url {
            scheme: scheme,
            user: user,
            host: host,
            port: port,
            path: path,
            query: query,
            fragment: fragment,
        }
    }
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
}

impl UserInfo {
    #[inline]
    pub fn new(user: String, pass: Option<String>) -> UserInfo {
        UserInfo { user: user, pass: pass }
    }
}

fn encode_inner(s: &str, full_url: bool) -> String {
    let mut rdr = BufReader::new(s.as_bytes());
    let mut out = String::new();

    loop {
        let mut buf = [0];
        let ch = match rdr.read(buf) {
            Err(..) => break,
            Ok(..) => buf[0] as char,
        };

        match ch {
          // unreserved:
          'A' .. 'Z' |
          'a' .. 'z' |
          '0' .. '9' |
          '-' | '.' | '_' | '~' => {
            out.push_char(ch);
          }
          _ => {
              if full_url {
                match ch {
                  // gen-delims:
                  ':' | '/' | '?' | '#' | '[' | ']' | '@' |

                  // sub-delims:
                  '!' | '$' | '&' | '"' | '(' | ')' | '*' |
                  '+' | ',' | ';' | '=' => {
                    out.push_char(ch);
                  }

                  _ => out.push_str(format!("%{:X}", ch as uint).as_slice())
                }
            } else {
                out.push_str(format!("%{:X}", ch as uint).as_slice());
            }
          }
        }
    }

    out
}

/**
 * Encodes a URI by replacing reserved characters with percent-encoded
 * character sequences.
 *
 * This function is compliant with RFC 3986.
 *
 * # Example
 *
 * ```rust
 * use url::encode;
 *
 * let url = encode("https://example.com/Rust (programming language)");
 * println!("{}", url); // https://example.com/Rust%20(programming%20language)
 * ```
 */
pub fn encode(s: &str) -> String {
    encode_inner(s, true)
}

/**
 * Encodes a URI component by replacing reserved characters with percent-
 * encoded character sequences.
 *
 * This function is compliant with RFC 3986.
 */

pub fn encode_component(s: &str) -> String {
    encode_inner(s, false)
}

fn decode_inner(s: &str, full_url: bool) -> String {
    let mut rdr = BufReader::new(s.as_bytes());
    let mut out = String::new();

    loop {
        let mut buf = [0];
        let ch = match rdr.read(buf) {
            Err(..) => break,
            Ok(..) => buf[0] as char
        };
        match ch {
          '%' => {
            let mut bytes = [0, 0];
            match rdr.read(bytes) {
                Ok(2) => {}
                _ => fail!() // FIXME: malformed url?
            }
            let ch = uint::parse_bytes(bytes, 16u).unwrap() as u8 as char;

            if full_url {
                // Only decode some characters:
                match ch {
                  // gen-delims:
                  ':' | '/' | '?' | '#' | '[' | ']' | '@' |

                  // sub-delims:
                  '!' | '$' | '&' | '"' | '(' | ')' | '*' |
                  '+' | ',' | ';' | '=' => {
                    out.push_char('%');
                    out.push_char(bytes[0u] as char);
                    out.push_char(bytes[1u] as char);
                  }

                  ch => out.push_char(ch)
                }
            } else {
                  out.push_char(ch);
            }
          }
          ch => out.push_char(ch)
        }
    }

    out
}

/**
 * Decodes a percent-encoded string representing a URI.
 *
 * This will only decode escape sequences generated by `encode`.
 *
 * # Example
 *
 * ```rust
 * use url::decode;
 *
 * let url = decode("https://example.com/Rust%20(programming%20language)");
 * println!("{}", url); // https://example.com/Rust (programming language)
 * ```
 */
pub fn decode(s: &str) -> String {
    decode_inner(s, true)
}

/**
 * Decode a string encoded with percent encoding.
 */
pub fn decode_component(s: &str) -> String {
    decode_inner(s, false)
}

fn encode_plus(s: &str) -> String {
    let mut rdr = BufReader::new(s.as_bytes());
    let mut out = String::new();

    loop {
        let mut buf = [0];
        let ch = match rdr.read(buf) {
            Ok(..) => buf[0] as char,
            Err(..) => break,
        };
        match ch {
          'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '_' | '.' | '-' => {
            out.push_char(ch);
          }
          ' ' => out.push_char('+'),
          _ => out.push_str(format!("%{:X}", ch as uint).as_slice())
        }
    }

    out
}

/**
 * Encode a hashmap to the 'application/x-www-form-urlencoded' media type.
 */
pub fn encode_form_urlencoded(m: &HashMap<String, Vec<String>>) -> String {
    let mut out = String::new();
    let mut first = true;

    for (key, values) in m.iter() {
        let key = encode_plus(key.as_slice());

        for value in values.iter() {
            if first {
                first = false;
            } else {
                out.push_char('&');
                first = false;
            }

            out.push_str(format!("{}={}",
                                 key,
                                 encode_plus(value.as_slice())).as_slice());
        }
    }

    out
}

/**
 * Decode a string encoded with the 'application/x-www-form-urlencoded' media
 * type into a hashmap.
 */
#[allow(experimental)]
pub fn decode_form_urlencoded(s: &[u8]) -> HashMap<String, Vec<String>> {
    let mut rdr = BufReader::new(s);
    let mut m: HashMap<String,Vec<String>> = HashMap::new();
    let mut key = String::new();
    let mut value = String::new();
    let mut parsing_key = true;

    loop {
        let mut buf = [0];
        let ch = match rdr.read(buf) {
            Ok(..) => buf[0] as char,
            Err(..) => break,
        };
        match ch {
            '&' | ';' => {
                if key.len() > 0 && value.len() > 0 {
                    let mut values = match m.pop_equiv(&key.as_slice()) {
                        Some(values) => values,
                        None => vec!(),
                    };

                    values.push(value);
                    m.insert(key, values);
                }

                parsing_key = true;
                key = String::new();
                value = String::new();
            }
            '=' => parsing_key = false,
            ch => {
                let ch = match ch {
                    '%' => {
                        let mut bytes = [0, 0];
                        match rdr.read(bytes) {
                            Ok(2) => {}
                            _ => fail!() // FIXME: malformed?
                        }
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
        }
    }

    if key.len() > 0 && value.len() > 0 {
        let mut values = match m.pop_equiv(&key.as_slice()) {
            Some(values) => values,
            None => vec!(),
        };

        values.push(value);
        m.insert(key, values);
    }

    m
}


fn split_char_first(s: &str, c: char) -> (String, String) {
    let len = s.len();
    let mut index = len;
    let mut mat = 0;
    let mut rdr = BufReader::new(s.as_bytes());
    loop {
        let mut buf = [0];
        let ch = match rdr.read(buf) {
            Ok(..) => buf[0] as char,
            Err(..) => break,
        };
        if ch == c {
            // found a match, adjust markers
            index = (rdr.tell().unwrap() as uint) - 1;
            mat = 1;
            break;
        }
    }
    if index+mat == len {
        return (s.slice(0, index).to_string(), "".to_string());
    } else {
        return (s.slice(0, index).to_string(),
                s.slice(index + mat, s.len()).to_string());
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

fn query_from_str(rawquery: &str) -> Query {
    let mut query: Query = vec!();
    if !rawquery.is_empty() {
        for p in rawquery.split('&') {
            let (k, v) = split_char_first(p, '=');
            query.push((decode_component(k.as_slice()),
                        decode_component(v.as_slice())));
        };
    }
    return query;
}

/**
 * Converts an instance of a URI `Query` type to a string.
 *
 * # Example
 *
 * ```rust
 * let query = vec!(("title".to_string(), "The Village".to_string()),
                    ("north".to_string(), "52.91".to_string()),
                    ("west".to_string(), "4.10".to_string()));
 * println!("{}", url::query_to_str(&query));  // title=The%20Village&north=52.91&west=4.10
 * ```
 */
#[allow(unused_must_use)]
pub fn query_to_str(query: &Query) -> String {
    use std::io::MemWriter;
    use std::str;

    let mut writer = MemWriter::new();
    for (i, &(ref k, ref v)) in query.iter().enumerate() {
        if i != 0 { write!(&mut writer, "&"); }
        write!(&mut writer, "{}={}", encode_component(k.as_slice()),
               encode_component(v.as_slice()));
    }
    str::from_utf8_lossy(writer.unwrap().as_slice()).to_string()
}

/**
 * Returns a tuple of the URI scheme and the rest of the URI, or a parsing error.
 *
 * Does not include the separating `:` character.
 *
 * # Example
 *
 * ```rust
 * use url::get_scheme;
 *
 * let scheme = match get_scheme("https://example.com/") {
 *     Ok((sch, _)) => sch,
 *     Err(_) => "(None)".to_string(),
 * };
 * println!("Scheme in use: {}.", scheme); // Scheme in use: https.
 * ```
 */
pub fn get_scheme(rawurl: &str) -> Result<(String, String), String> {
    for (i,c) in rawurl.chars().enumerate() {
        match c {
          'A' .. 'Z' | 'a' .. 'z' => continue,
          '0' .. '9' | '+' | '-' | '.' => {
            if i == 0 {
                return Err("url: Scheme must begin with a \
                            letter.".to_string());
            }
            continue;
          }
          ':' => {
            if i == 0 {
                return Err("url: Scheme cannot be empty.".to_string());
            } else {
                return Ok((rawurl.slice(0,i).to_string(),
                           rawurl.slice(i+1,rawurl.len()).to_string()));
            }
          }
          _ => {
            return Err("url: Invalid character in scheme.".to_string());
          }
        }
    };
    return Err("url: Scheme must be terminated with a colon.".to_string());
}

#[deriving(Clone, Eq)]
enum Input {
    Digit, // all digits
    Hex, // digits and letters a-f
    Unreserved // all other legal characters
}

// returns userinfo, host, port, and unparsed part, or an error
fn get_authority(rawurl: &str) ->
    Result<(Option<UserInfo>, String, Option<String>, String), String> {
    if !rawurl.starts_with("//") {
        // there is no authority.
        return Ok((None, "".to_string(), None, rawurl.to_str().to_string()));
    }

    enum State {
        Start, // starting state
        PassHostPort, // could be in user or port
        Ip6Port, // either in ipv6 host or port
        Ip6Host, // are in an ipv6 host
        InHost, // are in a host - may be ipv6, but don't know yet
        InPort // are in port
    }

    let len = rawurl.len();
    let mut st = Start;
    let mut input = Digit; // most restricted, start here.

    let mut userinfo = None;
    let mut host = "".to_string();
    let mut port = None;

    let mut colon_count = 0;
    let mut pos = 0;
    let mut begin = 2;
    let mut end = len;

    for (i,c) in rawurl.chars().enumerate() {
        if i < 2 { continue; } // ignore the leading //

        // deal with input class first
        match c {
          '0' .. '9' => (),
          'A' .. 'F' | 'a' .. 'f' => {
            if input == Digit {
                input = Hex;
            }
          }
          'G' .. 'Z' | 'g' .. 'z' | '-' | '.' | '_' | '~' | '%' |
          '&' |'\'' | '(' | ')' | '+' | '!' | '*' | ',' | ';' | '=' => {
            input = Unreserved;
          }
          ':' | '@' | '?' | '#' | '/' => {
            // separators, don't change anything
          }
          _ => {
            return Err("Illegal character in authority".to_string());
          }
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
                    host = rawurl.slice(begin, i).to_string();
                    st = InPort;
                } else {
                    // can't be sure whether this is an ipv6 address or a port
                    st = Ip6Port;
                }
              }
              Ip6Port => {
                if input == Unreserved {
                    return Err("Illegal characters in \
                                authority.".to_string());
                }
                st = Ip6Host;
              }
              Ip6Host => {
                if colon_count > 7 {
                    host = rawurl.slice(begin, i).to_string();
                    pos = i;
                    st = InPort;
                }
              }
              _ => {
                return Err("Invalid ':' in authority.".to_string());
              }
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
              _ => {
                return Err("Invalid '@' in authority.".to_string());
              }
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
      Start => {
        host = rawurl.slice(begin, end).to_string();
      }
      PassHostPort | Ip6Port => {
        if input != Digit {
            return Err("Non-digit characters in port.".to_string());
        }
        host = rawurl.slice(begin, pos).to_string();
        port = Some(rawurl.slice(pos+1, end).to_string());
      }
      Ip6Host | InHost => {
        host = rawurl.slice(begin, end).to_string();
      }
      InPort => {
        if input != Digit {
            return Err("Non-digit characters in port.".to_string());
        }
        port = Some(rawurl.slice(pos+1, end).to_string());
      }
    }

    let rest = rawurl.slice(end, len).to_string();
    return Ok((userinfo, host, port, rest));
}


// returns the path and unparsed part of url, or an error
fn get_path(rawurl: &str, authority: bool) ->
    Result<(String, String), String> {
    let len = rawurl.len();
    let mut end = len;
    for (i,c) in rawurl.chars().enumerate() {
        match c {
          'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '&' |'\'' | '(' | ')' | '.'
          | '@' | ':' | '%' | '/' | '+' | '!' | '*' | ',' | ';' | '='
          | '_' | '-' | '~' => {
            continue;
          }
          '?' | '#' => {
            end = i;
            break;
          }
          _ => return Err("Invalid character in path.".to_string())
        }
    }

    if authority {
        if end != 0 && !rawurl.starts_with("/") {
            return Err("Non-empty path must begin with\
                              '/' in presence of authority.".to_string());
        }
    }

    return Ok((decode_component(rawurl.slice(0, end)),
                    rawurl.slice(end, len).to_string()));
}

// returns the parsed query and the fragment, if present
fn get_query_fragment(rawurl: &str) ->
    Result<(Query, Option<String>), String> {
    if !rawurl.starts_with("?") {
        if rawurl.starts_with("#") {
            let f = decode_component(rawurl.slice(
                                                1,
                                                rawurl.len()));
            return Ok((vec!(), Some(f)));
        } else {
            return Ok((vec!(), None));
        }
    }
    let (q, r) = split_char_first(rawurl.slice(1, rawurl.len()), '#');
    let f = if r.len() != 0 {
        Some(decode_component(r.as_slice()))
    } else {
        None
    };
    return Ok((query_from_str(q.as_slice()), f));
}

/**
 * Parses a URL, converting it from a string to `Url` representation.
 *
 * # Arguments
 *
 * `rawurl` - a string representing the full URL, including scheme.
 *
 * # Returns
 *
 * A `Url` struct type representing the URL.
 */
pub fn from_str(rawurl: &str) -> Result<Url, String> {
    // scheme
    let (scheme, rest) = match get_scheme(rawurl) {
        Ok(val) => val,
        Err(e) => return Err(e),
    };

    // authority
    let (userinfo, host, port, rest) = match get_authority(rest.as_slice()) {
        Ok(val) => val,
        Err(e) => return Err(e),
    };

    // path
    let has_authority = host.len() > 0;
    let (path, rest) = match get_path(rest.as_slice(), has_authority) {
        Ok(val) => val,
        Err(e) => return Err(e),
    };

    // query and fragment
    let (query, fragment) = match get_query_fragment(rest.as_slice()) {
        Ok(val) => val,
        Err(e) => return Err(e),
    };

    Ok(Url::new(scheme, userinfo, host, port, path, query, fragment))
}

pub fn path_from_str(rawpath: &str) -> Result<Path, String> {
    let (path, rest) = match get_path(rawpath, false) {
        Ok(val) => val,
        Err(e) => return Err(e)
    };

    // query and fragment
    let (query, fragment) = match get_query_fragment(rest.as_slice()) {
        Ok(val) => val,
        Err(e) => return Err(e),
    };

    Ok(Path{ path: path, query: query, fragment: fragment })
}

impl FromStr for Url {
    fn from_str(s: &str) -> Option<Url> {
        match from_str(s) {
            Ok(url) => Some(url),
            Err(_) => None
        }
    }
}

impl FromStr for Path {
    fn from_str(s: &str) -> Option<Path> {
        match path_from_str(s) {
            Ok(path) => Some(path),
            Err(_) => None
        }
    }
}

impl fmt::Show for Url {
    /**
     * Converts a URL from `Url` to string representation.
     *
     * # Arguments
     *
     * `url` - a URL.
     *
     * # Returns
     *
     * A string that contains the formatted URL. Note that this will usually
     * be an inverse of `from_str` but might strip out unneeded separators;
     * for example, "http://somehost.com?", when parsed and formatted, will
     * result in just "http://somehost.com".
     */
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

        try!(write!(f, "{}", self.path));

        if !self.query.is_empty() {
            try!(write!(f, "?{}", query_to_str(&self.query)));
        }

        match self.fragment {
            Some(ref fragment) => {
                write!(f, "\\#{}", encode_component(fragment.as_slice()))
            }
            None => Ok(()),
        }
    }
}

impl fmt::Show for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{}", self.path));
        if !self.query.is_empty() {
            try!(write!(f, "?{}", self.query))
        }

        match self.fragment {
            Some(ref fragment) => {
                write!(f, "\\#{}", encode_component(fragment.as_slice()))
            }
            None => Ok(())
        }
    }
}

impl<S: Writer> Hash<S> for Url {
    fn hash(&self, state: &mut S) {
        self.to_str().hash(state)
    }
}

impl<S: Writer> Hash<S> for Path {
    fn hash(&self, state: &mut S) {
        self.to_str().hash(state)
    }
}

// Put a few tests outside of the 'test' module so they can test the internal
// functions and those functions don't need 'pub'

#[test]
fn test_split_char_first() {
    let (u,v) = split_char_first("hello, sweet world", ',');
    assert_eq!(u, "hello".to_string());
    assert_eq!(v, " sweet world".to_string());

    let (u,v) = split_char_first("hello sweet world", ',');
    assert_eq!(u, "hello sweet world".to_string());
    assert_eq!(v, "".to_string());
}

#[test]
fn test_get_authority() {
    let (u, h, p, r) = get_authority(
        "//user:pass@rust-lang.org/something").unwrap();
    assert_eq!(u, Some(UserInfo::new("user".to_string(), Some("pass".to_string()))));
    assert_eq!(h, "rust-lang.org".to_string());
    assert!(p.is_none());
    assert_eq!(r, "/something".to_string());

    let (u, h, p, r) = get_authority(
        "//rust-lang.org:8000?something").unwrap();
    assert!(u.is_none());
    assert_eq!(h, "rust-lang.org".to_string());
    assert_eq!(p, Some("8000".to_string()));
    assert_eq!(r, "?something".to_string());

    let (u, h, p, r) = get_authority(
        "//rust-lang.org#blah").unwrap();
    assert!(u.is_none());
    assert_eq!(h, "rust-lang.org".to_string());
    assert!(p.is_none());
    assert_eq!(r, "#blah".to_string());

    // ipv6 tests
    let (_, h, _, _) = get_authority(
        "//2001:0db8:85a3:0042:0000:8a2e:0370:7334#blah").unwrap();
    assert_eq!(h, "2001:0db8:85a3:0042:0000:8a2e:0370:7334".to_string());

    let (_, h, p, _) = get_authority(
        "//2001:0db8:85a3:0042:0000:8a2e:0370:7334:8000#blah").unwrap();
    assert_eq!(h, "2001:0db8:85a3:0042:0000:8a2e:0370:7334".to_string());
    assert_eq!(p, Some("8000".to_string()));

    let (u, h, p, _) = get_authority(
        "//us:p@2001:0db8:85a3:0042:0000:8a2e:0370:7334:8000#blah"
    ).unwrap();
    assert_eq!(u, Some(UserInfo::new("us".to_string(), Some("p".to_string()))));
    assert_eq!(h, "2001:0db8:85a3:0042:0000:8a2e:0370:7334".to_string());
    assert_eq!(p, Some("8000".to_string()));

    // invalid authorities;
    assert!(get_authority("//user:pass@rust-lang:something").is_err());
    assert!(get_authority("//user@rust-lang:something:/path").is_err());
    assert!(get_authority(
        "//2001:0db8:85a3:0042:0000:8a2e:0370:7334:800a").is_err());
    assert!(get_authority(
        "//2001:0db8:85a3:0042:0000:8a2e:0370:7334:8000:00").is_err());

    // these parse as empty, because they don't start with '//'
    let (_, h, _, _) = get_authority("user:pass@rust-lang").unwrap();
    assert_eq!(h, "".to_string());
    let (_, h, _, _) = get_authority("rust-lang.org").unwrap();
    assert_eq!(h, "".to_string());
}

#[test]
fn test_get_path() {
    let (p, r) = get_path("/something+%20orother", true).unwrap();
    assert_eq!(p, "/something+ orother".to_string());
    assert_eq!(r, "".to_string());
    let (p, r) = get_path("test@email.com#fragment", false).unwrap();
    assert_eq!(p, "test@email.com".to_string());
    assert_eq!(r, "#fragment".to_string());
    let (p, r) = get_path("/gen/:addr=?q=v", false).unwrap();
    assert_eq!(p, "/gen/:addr=".to_string());
    assert_eq!(r, "?q=v".to_string());

    //failure cases
    assert!(get_path("something?q", true).is_err());
}

#[cfg(test)]
mod tests {
    use {encode_form_urlencoded, decode_form_urlencoded,
         decode, encode, from_str, encode_component, decode_component,
         path_from_str, UserInfo, get_scheme};

    use collections::HashMap;

    #[test]
    fn test_url_parse() {
        let url = "http://user:pass@rust-lang.org:8080/doc/~u?s=v#something";

        let up = from_str(url);
        let u = up.unwrap();
        assert_eq!(&u.scheme, &"http".to_string());
        assert_eq!(&u.user, &Some(UserInfo::new("user".to_string(), Some("pass".to_string()))));
        assert_eq!(&u.host, &"rust-lang.org".to_string());
        assert_eq!(&u.port, &Some("8080".to_string()));
        assert_eq!(&u.path, &"/doc/~u".to_string());
        assert_eq!(&u.query, &vec!(("s".to_string(), "v".to_string())));
        assert_eq!(&u.fragment, &Some("something".to_string()));
    }

    #[test]
    fn test_path_parse() {
        let path = "/doc/~u?s=v#something";

        let up = path_from_str(path);
        let u = up.unwrap();
        assert_eq!(&u.path, &"/doc/~u".to_string());
        assert_eq!(&u.query, &vec!(("s".to_string(), "v".to_string())));
        assert_eq!(&u.fragment, &Some("something".to_string()));
    }

    #[test]
    fn test_url_parse_host_slash() {
        let urlstr = "http://0.42.42.42/";
        let url = from_str(urlstr).unwrap();
        assert!(url.host == "0.42.42.42".to_string());
        assert!(url.path == "/".to_string());
    }

    #[test]
    fn test_path_parse_host_slash() {
        let pathstr = "/";
        let path = path_from_str(pathstr).unwrap();
        assert!(path.path == "/".to_string());
    }

    #[test]
    fn test_url_host_with_port() {
        let urlstr = "scheme://host:1234";
        let url = from_str(urlstr).unwrap();
        assert_eq!(&url.scheme, &"scheme".to_string());
        assert_eq!(&url.host, &"host".to_string());
        assert_eq!(&url.port, &Some("1234".to_string()));
        // is empty path really correct? Other tests think so
        assert_eq!(&url.path, &"".to_string());
        let urlstr = "scheme://host:1234/";
        let url = from_str(urlstr).unwrap();
        assert_eq!(&url.scheme, &"scheme".to_string());
        assert_eq!(&url.host, &"host".to_string());
        assert_eq!(&url.port, &Some("1234".to_string()));
        assert_eq!(&url.path, &"/".to_string());
    }

    #[test]
    fn test_url_with_underscores() {
        let urlstr = "http://dotcom.com/file_name.html";
        let url = from_str(urlstr).unwrap();
        assert!(url.path == "/file_name.html".to_string());
    }

    #[test]
    fn test_path_with_underscores() {
        let pathstr = "/file_name.html";
        let path = path_from_str(pathstr).unwrap();
        assert!(path.path == "/file_name.html".to_string());
    }

    #[test]
    fn test_url_with_dashes() {
        let urlstr = "http://dotcom.com/file-name.html";
        let url = from_str(urlstr).unwrap();
        assert!(url.path == "/file-name.html".to_string());
    }

    #[test]
    fn test_path_with_dashes() {
        let pathstr = "/file-name.html";
        let path = path_from_str(pathstr).unwrap();
        assert!(path.path == "/file-name.html".to_string());
    }

    #[test]
    fn test_no_scheme() {
        assert!(get_scheme("noschemehere.html").is_err());
    }

    #[test]
    fn test_invalid_scheme_errors() {
        assert!(from_str("99://something").is_err());
        assert!(from_str("://something").is_err());
    }

    #[test]
    fn test_full_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org/doc?s=v#something";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_userless_url_parse_and_format() {
        let url = "http://rust-lang.org/doc?s=v#something";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_queryless_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org/doc#something";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_empty_query_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org/doc?#something";
        let should_be = "http://user:pass@rust-lang.org/doc#something";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), should_be);
    }

    #[test]
    fn test_fragmentless_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org/doc?q=v";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_minimal_url_parse_and_format() {
        let url = "http://rust-lang.org/doc";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_url_with_port_parse_and_format() {
        let url = "http://rust-lang.org:80/doc";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_scheme_host_only_url_parse_and_format() {
        let url = "http://rust-lang.org";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_pathless_url_parse_and_format() {
        let url = "http://user:pass@rust-lang.org?q=v#something";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_scheme_host_fragment_only_url_parse_and_format() {
        let url = "http://rust-lang.org#something";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_url_component_encoding() {
        let url = "http://rust-lang.org/doc%20uments?ba%25d%20=%23%26%2B";
        let u = from_str(url).unwrap();
        assert!(u.path == "/doc uments".to_string());
        assert!(u.query == vec!(("ba%d ".to_string(), "#&+".to_string())));
    }

    #[test]
    fn test_path_component_encoding() {
        let path = "/doc%20uments?ba%25d%20=%23%26%2B";
        let p = path_from_str(path).unwrap();
        assert!(p.path == "/doc uments".to_string());
        assert!(p.query == vec!(("ba%d ".to_string(), "#&+".to_string())));
    }

    #[test]
    fn test_url_without_authority() {
        let url = "mailto:test@email.com";
        assert_eq!(from_str(url).unwrap().to_str().as_slice(), url);
    }

    #[test]
    fn test_encode() {
        assert_eq!(encode(""), "".to_string());
        assert_eq!(encode("http://example.com"), "http://example.com".to_string());
        assert_eq!(encode("foo bar% baz"), "foo%20bar%25%20baz".to_string());
        assert_eq!(encode(" "), "%20".to_string());
        assert_eq!(encode("!"), "!".to_string());
        assert_eq!(encode("\""), "\"".to_string());
        assert_eq!(encode("#"), "#".to_string());
        assert_eq!(encode("$"), "$".to_string());
        assert_eq!(encode("%"), "%25".to_string());
        assert_eq!(encode("&"), "&".to_string());
        assert_eq!(encode("'"), "%27".to_string());
        assert_eq!(encode("("), "(".to_string());
        assert_eq!(encode(")"), ")".to_string());
        assert_eq!(encode("*"), "*".to_string());
        assert_eq!(encode("+"), "+".to_string());
        assert_eq!(encode(","), ",".to_string());
        assert_eq!(encode("/"), "/".to_string());
        assert_eq!(encode(":"), ":".to_string());
        assert_eq!(encode(";"), ";".to_string());
        assert_eq!(encode("="), "=".to_string());
        assert_eq!(encode("?"), "?".to_string());
        assert_eq!(encode("@"), "@".to_string());
        assert_eq!(encode("["), "[".to_string());
        assert_eq!(encode("]"), "]".to_string());
    }

    #[test]
    fn test_encode_component() {
        assert_eq!(encode_component(""), "".to_string());
        assert!(encode_component("http://example.com") ==
            "http%3A%2F%2Fexample.com".to_string());
        assert!(encode_component("foo bar% baz") ==
            "foo%20bar%25%20baz".to_string());
        assert_eq!(encode_component(" "), "%20".to_string());
        assert_eq!(encode_component("!"), "%21".to_string());
        assert_eq!(encode_component("#"), "%23".to_string());
        assert_eq!(encode_component("$"), "%24".to_string());
        assert_eq!(encode_component("%"), "%25".to_string());
        assert_eq!(encode_component("&"), "%26".to_string());
        assert_eq!(encode_component("'"), "%27".to_string());
        assert_eq!(encode_component("("), "%28".to_string());
        assert_eq!(encode_component(")"), "%29".to_string());
        assert_eq!(encode_component("*"), "%2A".to_string());
        assert_eq!(encode_component("+"), "%2B".to_string());
        assert_eq!(encode_component(","), "%2C".to_string());
        assert_eq!(encode_component("/"), "%2F".to_string());
        assert_eq!(encode_component(":"), "%3A".to_string());
        assert_eq!(encode_component(";"), "%3B".to_string());
        assert_eq!(encode_component("="), "%3D".to_string());
        assert_eq!(encode_component("?"), "%3F".to_string());
        assert_eq!(encode_component("@"), "%40".to_string());
        assert_eq!(encode_component("["), "%5B".to_string());
        assert_eq!(encode_component("]"), "%5D".to_string());
    }

    #[test]
    fn test_decode() {
        assert_eq!(decode(""), "".to_string());
        assert_eq!(decode("abc/def 123"), "abc/def 123".to_string());
        assert_eq!(decode("abc%2Fdef%20123"), "abc%2Fdef 123".to_string());
        assert_eq!(decode("%20"), " ".to_string());
        assert_eq!(decode("%21"), "%21".to_string());
        assert_eq!(decode("%22"), "%22".to_string());
        assert_eq!(decode("%23"), "%23".to_string());
        assert_eq!(decode("%24"), "%24".to_string());
        assert_eq!(decode("%25"), "%".to_string());
        assert_eq!(decode("%26"), "%26".to_string());
        assert_eq!(decode("%27"), "'".to_string());
        assert_eq!(decode("%28"), "%28".to_string());
        assert_eq!(decode("%29"), "%29".to_string());
        assert_eq!(decode("%2A"), "%2A".to_string());
        assert_eq!(decode("%2B"), "%2B".to_string());
        assert_eq!(decode("%2C"), "%2C".to_string());
        assert_eq!(decode("%2F"), "%2F".to_string());
        assert_eq!(decode("%3A"), "%3A".to_string());
        assert_eq!(decode("%3B"), "%3B".to_string());
        assert_eq!(decode("%3D"), "%3D".to_string());
        assert_eq!(decode("%3F"), "%3F".to_string());
        assert_eq!(decode("%40"), "%40".to_string());
        assert_eq!(decode("%5B"), "%5B".to_string());
        assert_eq!(decode("%5D"), "%5D".to_string());
    }

    #[test]
    fn test_decode_component() {
        assert_eq!(decode_component(""), "".to_string());
        assert_eq!(decode_component("abc/def 123"), "abc/def 123".to_string());
        assert_eq!(decode_component("abc%2Fdef%20123"), "abc/def 123".to_string());
        assert_eq!(decode_component("%20"), " ".to_string());
        assert_eq!(decode_component("%21"), "!".to_string());
        assert_eq!(decode_component("%22"), "\"".to_string());
        assert_eq!(decode_component("%23"), "#".to_string());
        assert_eq!(decode_component("%24"), "$".to_string());
        assert_eq!(decode_component("%25"), "%".to_string());
        assert_eq!(decode_component("%26"), "&".to_string());
        assert_eq!(decode_component("%27"), "'".to_string());
        assert_eq!(decode_component("%28"), "(".to_string());
        assert_eq!(decode_component("%29"), ")".to_string());
        assert_eq!(decode_component("%2A"), "*".to_string());
        assert_eq!(decode_component("%2B"), "+".to_string());
        assert_eq!(decode_component("%2C"), ",".to_string());
        assert_eq!(decode_component("%2F"), "/".to_string());
        assert_eq!(decode_component("%3A"), ":".to_string());
        assert_eq!(decode_component("%3B"), ";".to_string());
        assert_eq!(decode_component("%3D"), "=".to_string());
        assert_eq!(decode_component("%3F"), "?".to_string());
        assert_eq!(decode_component("%40"), "@".to_string());
        assert_eq!(decode_component("%5B"), "[".to_string());
        assert_eq!(decode_component("%5D"), "]".to_string());
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
        assert!(encode_form_urlencoded(&m) ==
            "foo+bar=abc&foo+bar=12+%3D+34".to_string());
    }

    #[test]
    fn test_decode_form_urlencoded() {
        assert_eq!(decode_form_urlencoded([]).len(), 0);

        let s = "a=1&foo+bar=abc&foo+bar=12+%3D+34".as_bytes();
        let form = decode_form_urlencoded(s);
        assert_eq!(form.len(), 2);
        assert_eq!(form.get(&"a".to_string()), &vec!("1".to_string()));
        assert_eq!(form.get(&"foo bar".to_string()),
                   &vec!("abc".to_string(), "12 = 34".to_string()));
    }
}
