/*!

Cross-platform file path handling

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::Eq;

pub struct WindowsPath {
    host: Option<~str>,
    device: Option<~str>,
    is_absolute: bool,
    components: ~[~str],
}

pub struct PosixPath {
    is_absolute: bool,
    components: ~[~str],
}

pub trait GenericPath {

    static pure fn from_str((&str)) -> self;

    pure fn dirname() -> ~str;
    pure fn filename() -> Option<~str>;
    pure fn filestem() -> Option<~str>;
    pure fn filetype() -> Option<~str>;

    pure fn with_dirname((&str)) -> self;
    pure fn with_filename((&str)) -> self;
    pure fn with_filestem((&str)) -> self;
    pure fn with_filetype((&str)) -> self;

    pure fn push((&str)) -> self;
    pure fn push_rel((&self)) -> self;
    pure fn push_many((&[~str])) -> self;
    pure fn pop() -> self;

    pure fn normalize() -> self;
}

#[cfg(windows)]
pub type Path = WindowsPath;

#[cfg(windows)]
pub pure fn Path(s: &str) -> Path {
    from_str::<WindowsPath>(s)
}

#[cfg(unix)]
pub type Path = PosixPath;

#[cfg(unix)]
pub pure fn Path(s: &str) -> Path {
    from_str::<PosixPath>(s)
}

impl PosixPath : ToStr {
    fn to_str() -> ~str {
        let mut s = ~"";
        if self.is_absolute {
            s += "/";
        }
        s + str::connect(self.components, "/")
    }
}

impl PosixPath : Eq {
    pure fn eq(other: &PosixPath) -> bool {
        return self.is_absolute == (*other).is_absolute &&
            self.components == (*other).components;
    }
    pure fn ne(other: &PosixPath) -> bool { !self.eq(other) }
}

impl WindowsPath : Eq {
    pure fn eq(other: &WindowsPath) -> bool {
        return self.host == (*other).host &&
            self.device == (*other).device &&
            self.is_absolute == (*other).is_absolute &&
            self.components == (*other).components;
    }
    pure fn ne(other: &WindowsPath) -> bool { !self.eq(other) }
}

// FIXME (#3227): when default methods in traits are working, de-duplicate
// PosixPath and WindowsPath, most of their methods are common.
impl PosixPath : GenericPath {

    static pure fn from_str(s: &str) -> PosixPath {
        let mut components = str::split_nonempty(s, |c| c == '/');
        let is_absolute = (s.len() != 0 && s[0] == '/' as u8);
        return PosixPath { is_absolute: is_absolute,
                           components: components }
    }

    pure fn dirname() -> ~str {
        unsafe {
            let s = self.dir_path().to_str();
            if s.len() == 0 {
                ~"."
            } else {
                move s
            }
        }
    }

    pure fn filename() -> Option<~str> {
        match self.components.len() {
          0 => None,
          n => Some(copy self.components[n - 1])
        }
    }

    pure fn filestem() -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) => Some(f.slice(0, p)),
              None => Some(copy *f)
            }
          }
        }
    }

    pure fn filetype() -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) if p < f.len() => Some(f.slice(p, f.len())),
              _ => None
            }
          }
        }
    }

    pure fn with_dirname(d: &str) -> PosixPath {
        let dpath = from_str::<PosixPath>(d);
        match self.filename() {
          Some(ref f) => dpath.push(*f),
          None => move dpath
        }
    }

    pure fn with_filename(f: &str) -> PosixPath {
        unsafe {
            assert ! str::any(f, |c| windows::is_sep(c as u8));
            self.dir_path().push(f)
        }
    }

    pure fn with_filestem(s: &str) -> PosixPath {
        match self.filetype() {
          None => self.with_filename(s),
          Some(ref t) => self.with_filename(str::from_slice(s) + *t)
        }
    }

    pure fn with_filetype(t: &str) -> PosixPath {
        if t.len() == 0 {
            match self.filestem() {
              None => copy self,
              Some(s) => self.with_filename(s)
            }
        } else {
            let t = ~"." + str::from_slice(t);
            match self.filestem() {
              None => self.with_filename(t),
              Some(ref s) => self.with_filename(*s + t)
            }
        }
    }

    pure fn dir_path() -> PosixPath {
        if self.components.len() != 0 {
            self.pop()
        } else {
            copy self
        }
    }

    pure fn file_path() -> PosixPath {
        let cs = match self.filename() {
          None => ~[],
          Some(ref f) => ~[copy *f]
        };
        return PosixPath { is_absolute: false,
                           components: cs }
    }

    pure fn push_rel(other: &PosixPath) -> PosixPath {
        assert !other.is_absolute;
        self.push_many(other.components)
    }

    pure fn push_many(cs: &[~str]) -> PosixPath {
        let mut v = copy self.components;
        for cs.each |e| {
            let mut ss = str::split_nonempty(
                *e,
                |c| windows::is_sep(c as u8));
            unsafe { v.push_all_move(move ss); }
        }
        PosixPath { components: move v, ..self }
    }

    pure fn push(s: &str) -> PosixPath {
        let mut v = copy self.components;
        let mut ss = str::split_nonempty(s, |c| windows::is_sep(c as u8));
        unsafe { v.push_all_move(move ss); }
        PosixPath { components: move v, ..self }
    }

    pure fn pop() -> PosixPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            unsafe { vec::pop(cs); }
        }
        return PosixPath { components: move cs, ..self }
    }

    pure fn normalize() -> PosixPath {
        return PosixPath {
            components: normalize(self.components),
            ..self
        }
    }
}


impl WindowsPath : ToStr {
    fn to_str() -> ~str {
        let mut s = ~"";
        match self.host {
          Some(h) => { s += "\\\\"; s += h; }
          None => { }
        }
        match self.device {
          Some(d) => { s += d; s += ":"; }
          None => { }
        }
        if self.is_absolute {
            s += "\\";
        }
        s + str::connect(self.components, "\\")
    }
}


impl WindowsPath : GenericPath {

    static pure fn from_str(s: &str) -> WindowsPath {
        let host;
        let device;
        let rest;

        match windows::extract_drive_prefix(s) {
          Some((ref d, ref r)) => {
            host = None;
            device = Some(copy *d);
            rest = copy *r;
          }
          None => {
            match windows::extract_unc_prefix(s) {
              Some((ref h, ref r)) => {
                host = Some(copy *h);
                device = None;
                rest = copy *r;
              }
              None => {
                host = None;
                device = None;
                rest = str::from_slice(s);
              }
            }
          }
        }

        let mut components =
            str::split_nonempty(rest, |c| windows::is_sep(c as u8));
        let is_absolute = (rest.len() != 0 && windows::is_sep(rest[0]));
        return WindowsPath { host: host,
                             device: device,
                             is_absolute: is_absolute,
                             components: components }
    }

    pure fn dirname() -> ~str {
        unsafe {
            let s = self.dir_path().to_str();
            if s.len() == 0 {
                ~"."
            } else {
                move s
            }
        }
    }

    pure fn filename() -> Option<~str> {
        match self.components.len() {
          0 => None,
          n => Some(copy self.components[n - 1])
        }
    }

    pure fn filestem() -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) => Some(f.slice(0, p)),
              None => Some(copy *f)
            }
          }
        }
    }

    pure fn filetype() -> Option<~str> {
        match self.filename() {
          None => None,
          Some(ref f) => {
            match str::rfind_char(*f, '.') {
              Some(p) if p < f.len() => Some(f.slice(p, f.len())),
              _ => None
            }
          }
        }
    }

    pure fn with_dirname(d: &str) -> WindowsPath {
        let dpath = from_str::<WindowsPath>(d);
        match self.filename() {
          Some(ref f) => dpath.push(*f),
          None => move dpath
        }
    }

    pure fn with_filename(f: &str) -> WindowsPath {
        assert ! str::any(f, |c| windows::is_sep(c as u8));
        self.dir_path().push(f)
    }

    pure fn with_filestem(s: &str) -> WindowsPath {
        match self.filetype() {
          None => self.with_filename(s),
          Some(ref t) => self.with_filename(str::from_slice(s) + *t)
        }
    }

    pure fn with_filetype(t: &str) -> WindowsPath {
        if t.len() == 0 {
            match self.filestem() {
              None => copy self,
              Some(s) => self.with_filename(s)
            }
        } else {
            let t = ~"." + str::from_slice(t);
            match self.filestem() {
              None => self.with_filename(t),
              Some(ref s) =>
              self.with_filename(*s + t)
            }
        }
    }

    pure fn dir_path() -> WindowsPath {
        if self.components.len() != 0 {
            self.pop()
        } else {
            copy self
        }
    }

    pure fn file_path() -> WindowsPath {
        let cs = match self.filename() {
          None => ~[],
          Some(ref f) => ~[copy *f]
        };
        return WindowsPath { host: None,
                             device: None,
                             is_absolute: false,
                             components: cs }
    }

    pure fn push_rel(other: &WindowsPath) -> WindowsPath {
        assert !other.is_absolute;
        self.push_many(other.components)
    }

    pure fn push_many(cs: &[~str]) -> WindowsPath {
        let mut v = copy self.components;
        for cs.each |e| {
            let mut ss = str::split_nonempty(
                *e,
                |c| windows::is_sep(c as u8));
            unsafe { v.push_all_move(move ss); }
        }
        return WindowsPath { components: move v, ..self }
    }

    pure fn push(s: &str) -> WindowsPath {
        let mut v = copy self.components;
        let mut ss = str::split_nonempty(s, |c| windows::is_sep(c as u8));
        unsafe { v.push_all_move(move ss); }
        return WindowsPath { components: move v, ..self }
    }

    pure fn pop() -> WindowsPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            unsafe { vec::pop(cs); }
        }
        return WindowsPath { components: move cs, ..self }
    }

    pure fn normalize() -> WindowsPath {
        return WindowsPath {
            components: normalize(self.components),
            ..self
        }
    }
}


pub pure fn normalize(components: &[~str]) -> ~[~str] {
    let mut cs = ~[];
    unsafe {
        for components.each |c| {
            unsafe {
                if *c == ~"." && components.len() > 1 { loop; }
                if *c == ~"" { loop; }
                if *c == ~".." && cs.len() != 0 {
                    vec::pop(cs);
                    loop;
                }
                cs.push(copy *c);
            }
        }
    }
    move cs
}

#[test]
fn test_double_slash_collapsing()
{
    let path = from_str::<PosixPath>("tmp/");
    let path = path.push("/hmm");
    let path = path.normalize();
    assert ~"tmp/hmm" == path.to_str();

    let path = from_str::<WindowsPath>("tmp/");
    let path = path.push("/hmm");
    let path = path.normalize();
    assert ~"tmp\\hmm" == path.to_str();
}

mod posix {

    #[cfg(test)]
    fn mk(s: &str) -> PosixPath { from_str::<PosixPath>(s) }

    #[cfg(test)]
    fn t(wp: &PosixPath, s: &str) {
        let ss = wp.to_str();
        let sss = str::from_slice(s);
        if (ss != sss) {
            debug!("got %s", ss);
            debug!("expected %s", sss);
            assert ss == sss;
        }
    }

    #[test]
    fn test_filetype_foo_bar() {
        let wp = mk("foo.bar");
        assert wp.filetype() == Some(~".bar");
    }

    #[test]
    fn test_filetype_foo() {
        let wp = mk("foo");
        assert wp.filetype() == None;
    }

    #[test]
    fn test_posix_paths() {
        t(&(mk("hi")), "hi");
        t(&(mk("/lib")), "/lib");
        t(&(mk("hi/there")), "hi/there");
        t(&(mk("hi/there.txt")), "hi/there.txt");

        t(&(mk("hi/there.txt")), "hi/there.txt");
        t(&(mk("hi/there.txt")
           .with_filetype("")), "hi/there");

        t(&(mk("/a/b/c/there.txt")
            .with_dirname("hi")), "hi/there.txt");

        t(&(mk("hi/there.txt")
            .with_dirname(".")), "./there.txt");

        t(&(mk("a/b/c")
            .push("..")), "a/b/c/..");

        t(&(mk("there.txt")
            .with_filetype("o")), "there.o");

        t(&(mk("hi/there.txt")
            .with_filetype("o")), "hi/there.o");

        t(&(mk("hi/there.txt")
            .with_filetype("o")
            .with_dirname("/usr/lib")),
          "/usr/lib/there.o");

        t(&(mk("hi/there.txt")
            .with_filetype("o")
            .with_dirname("/usr/lib/")),
          "/usr/lib/there.o");

        t(&(mk("hi/there.txt")
            .with_filetype("o")
            .with_dirname("/usr//lib//")),
            "/usr/lib/there.o");

        t(&(mk("/usr/bin/rust")
            .push_many([~"lib", ~"thingy.so"])
            .with_filestem("librustc")),
          "/usr/bin/rust/lib/librustc.so");

    }

    #[test]
    fn test_normalize() {
        t(&(mk("hi/there.txt")
            .with_dirname(".").normalize()), "there.txt");

        t(&(mk("a/b/../c/././/../foo.txt/").normalize()),
          "a/foo.txt");

        t(&(mk("a/b/c")
            .push("..").normalize()), "a/b");
    }
}

// Various windows helpers, and tests for the impl.
mod windows {

    #[inline(always)]
    pub pure fn is_sep(u: u8) -> bool {
        u == '/' as u8 || u == '\\' as u8
    }

    pub pure fn extract_unc_prefix(s: &str) -> Option<(~str,~str)> {
        if (s.len() > 1 &&
            s[0] == '\\' as u8 &&
            s[1] == '\\' as u8) {
            let mut i = 2;
            while i < s.len() {
                if s[i] == '\\' as u8 {
                    let pre = s.slice(2, i);
                    let rest = s.slice(i, s.len());
                    return Some((move pre, move rest));
                }
                i += 1;
            }
        }
        None
    }

    pub pure fn extract_drive_prefix(s: &str) -> Option<(~str,~str)> {
        unsafe {
            if (s.len() > 1 &&
                libc::isalpha(s[0] as libc::c_int) != 0 &&
                s[1] == ':' as u8) {
                let rest = if s.len() == 2 {
                    ~""
                } else {
                    s.slice(2, s.len())
                };
                return Some((s.slice(0,1), move rest));
            }
            None
        }
    }

    #[test]
    fn test_extract_unc_prefixes() {
        assert extract_unc_prefix("\\\\").is_none();
        assert extract_unc_prefix("\\\\hi").is_none();
        assert extract_unc_prefix("\\\\hi\\") == Some((~"hi", ~"\\"));
        assert extract_unc_prefix("\\\\hi\\there") ==
            Some((~"hi", ~"\\there"));
        assert extract_unc_prefix("\\\\hi\\there\\friends.txt") ==
            Some((~"hi", ~"\\there\\friends.txt"));
    }

    #[test]
    fn test_extract_drive_prefixes() {
        assert extract_drive_prefix("c").is_none();
        assert extract_drive_prefix("c:") == Some((~"c", ~""));
        assert extract_drive_prefix("d:") == Some((~"d", ~""));
        assert extract_drive_prefix("z:") == Some((~"z", ~""));
        assert extract_drive_prefix("c:\\hi") == Some((~"c", ~"\\hi"));
        assert extract_drive_prefix("d:hi") == Some((~"d", ~"hi"));
        assert extract_drive_prefix("c:hi\\there.txt") ==
            Some((~"c", ~"hi\\there.txt"));
        assert extract_drive_prefix("c:\\hi\\there.txt") ==
            Some((~"c", ~"\\hi\\there.txt"));
    }

    #[test]
    fn test_windows_paths() {
        fn mk(s: &str) -> WindowsPath { from_str::<WindowsPath>(s) }
        fn t(wp: &WindowsPath, s: &str) {
            let ss = wp.to_str();
            let sss = str::from_slice(s);
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert ss == sss;
            }
        }

        t(&(mk("hi")), "hi");
        t(&(mk("hi/there")), "hi\\there");
        t(&(mk("hi/there.txt")), "hi\\there.txt");

        t(&(mk("there.txt")
            .with_filetype("o")), "there.o");

        t(&(mk("hi/there.txt")
            .with_filetype("o")), "hi\\there.o");

        t(&(mk("hi/there.txt")
            .with_filetype("o")
            .with_dirname("c:\\program files A")),
          "c:\\program files A\\there.o");

        t(&(mk("hi/there.txt")
            .with_filetype("o")
            .with_dirname("c:\\program files B\\")),
          "c:\\program files B\\there.o");

        t(&(mk("hi/there.txt")
            .with_filetype("o")
            .with_dirname("c:\\program files C\\/")),
            "c:\\program files C\\there.o");

        t(&(mk("c:\\program files (x86)\\rust")
            .push_many([~"lib", ~"thingy.dll"])
            .with_filename("librustc.dll")),
          "c:\\program files (x86)\\rust\\lib\\librustc.dll");

    }

    #[cfg(test)]
    fn mk(s: &str) -> PosixPath { from_str::<PosixPath>(s) }

    #[test]
    fn test_filetype_foo_bar() {
        let wp = mk("foo.bar");
        assert wp.filetype() == Some(~".bar");
    }

    #[test]
    fn test_filetype_foo() {
        let wp = mk("foo");
        assert wp.filetype() == None;
    }

}
