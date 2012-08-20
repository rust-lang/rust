// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

struct WindowsPath {
    host: option<~str>;
    device: option<~str>;
    is_absolute: bool;
    components: ~[~str];
}

struct PosixPath {
    is_absolute: bool;
    components: ~[~str];
}

trait Path {

    static fn from_str((&str)) -> self;
    fn to_str() -> ~str;

    fn dirname() -> ~str;
    fn filename() -> option<~str>;
    fn filestem() -> option<~str>;
    fn filetype() -> option<~str>;

    fn with_dirname((&str)) -> self;
    fn with_filename((&str)) -> self;
    fn with_filestem((&str)) -> self;
    fn with_filetype((&str)) -> self;

    fn push_components((&[~str])) -> self;
    fn pop_component() -> self;
}

// FIXME (#3227): when default methods in traits are working, de-duplicate
// PosixPath and WindowsPath, most of their methods are common.

impl PosixPath : Path {
    fn to_str() -> ~str {
        match self.filename() {
          none => self.dirname(),
          some(ref f) =>
          if (self.components.len() == 1 &&
              !self.is_absolute) {
            copy *f
          } else {
            self.dirname() + "/" + *f
          }
        }
    }

    static fn from_str(s: &str) -> PosixPath {
        let mut components = str::split_nonempty(s, |c| c == '/');
        let is_absolute = (s.len() != 0 && s[0] == '/' as u8);
        return PosixPath { is_absolute: is_absolute,
                           components: normalize(components) }
    }

    fn dirname() -> ~str {
        let mut s = ~"";
        if self.is_absolute {
            s += "/";
        }
        let mut d = copy self.components;
        if d.len() != 0 {
            vec::pop(d);
        }
        s += str::connect(d, "/");
        if s.len() == 0 {
            s = ~".";
        }
        return s;
    }

    fn filename() -> option<~str> {
        match self.components.len() {
          0 => none,
          n => some(copy self.components[n - 1])
        }
    }

    fn filestem() -> option<~str> {
        match self.filename() {
          none => none,
          some(ref f) => {
            match str::rfind_char(*f, '.') {
              some(p) => some(f.slice(0, p)),
              none => some(copy *f)
            }
          }
        }
    }

    fn filetype() -> option<~str> {
        match self.filename() {
          none => none,
          some(ref f) => {
            match str::rfind_char(*f, '.') {
              some(p) if p+1 < f.len() => some(f.slice(p+1, f.len())),
              _ => none
            }
          }
        }
    }

    fn with_dirname(d: &str) -> PosixPath {
        let dpath = from_str::<PosixPath>(d);
        match self.filename() {
          some(ref f) => dpath.push_components(~[copy *f]),
          none => dpath
        }
    }

    fn with_filename(f: &str) -> PosixPath {
        assert ! str::any(f, |c| windows::is_sep(c as u8));
        self.dir_path().push_components(~[str::from_slice(f)])
    }

    fn with_filestem(s: &str) -> PosixPath {
        match self.filetype() {
          none => self.with_filename(s),
          some(ref t) =>
          self.with_filename(str::from_slice(s) + "." + *t)
        }
    }

    fn with_filetype(t: &str) -> PosixPath {
        if t.len() == 0 {
            match self.filestem() {
              none => copy self,
              some(s) => self.with_filename(s)
            }
        } else {
            let t = ~"." + str::from_slice(t);
            match self.filestem() {
              none => self.with_filename(t),
              some(ref s) =>
              self.with_filename(*s + t)
            }
        }
    }

    fn dir_path() -> PosixPath {
        if self.components.len() != 0 {
            self.pop_component()
        } else {
            copy self
        }
    }

    fn file_path() -> PosixPath {
        let cs = match self.filename() {
          none => ~[],
          some(ref f) => ~[copy *f]
        };
        return PosixPath { is_absolute: false,
                           components: cs }
    }

    fn push_components(cs: &[~str]) -> PosixPath {
        return PosixPath { components: normalize(self.components + cs),
                           ..self }
    }

    fn pop_component() -> PosixPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            vec::pop(cs);
        }
        return PosixPath { components: cs, ..self }
    }



}


impl WindowsPath : Path {

    fn to_str() -> ~str {
        match self.filename() {
          none => self.dirname(),
          some(ref f) =>
          if (self.components.len() == 1 &&
              !self.is_absolute &&
              self.host == none &&
              self.device == none) {
            copy *f
          } else {
            self.dirname() + "\\" + *f
          }
        }
    }

    static fn from_str(s: &str) -> WindowsPath {
        let host;
        let device;
        let rest;

        match windows::extract_drive_prefix(s) {
          some((ref d, ref r)) => {
            host = none;
            device = some(copy *d);
            rest = copy *r;
          }
          none => {
            match windows::extract_unc_prefix(s) {
              some((ref h, ref r)) => {
                host = some(copy *h);
                device = none;
                rest = copy *r;
              }
              none => {
                host = none;
                device = none;
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
                             components: normalize(components) }
    }

    fn dirname() -> ~str {
        let mut s = ~"";
        match self.host {
          some(h) => { s += "\\\\"; s += h; }
          none => { }
        }
        match self.device {
          some(d) => { s += d; s += ":"; }
          none => { }
        }
        if self.is_absolute {
            s += "\\";
        }
        let mut d = copy self.components;
        if d.len() != 0 {
            vec::pop(d);
        }
        s += str::connect(d, "\\");
        if s.len() == 0 {
            s = ~".";
        }
        return s;
    }

    fn filename() -> option<~str> {
        match self.components.len() {
          0 => none,
          n => some(copy self.components[n - 1])
        }
    }

    fn filestem() -> option<~str> {
        match self.filename() {
          none => none,
          some(ref f) => {
            match str::rfind_char(*f, '.') {
              some(p) => some(f.slice(0, p)),
              none => some(copy *f)
            }
          }
        }
    }

    fn filetype() -> option<~str> {
        match self.filename() {
          none => none,
          some(ref f) => {
            match str::rfind_char(*f, '.') {
              some(p) if p+1 < f.len() => some(f.slice(p+1, f.len())),
              _ => none
            }
          }
        }
    }

    fn with_dirname(d: &str) -> WindowsPath {
        let dpath = from_str::<WindowsPath>(d);
        match self.filename() {
          some(ref f) => dpath.push_components(~[copy *f]),
          none => dpath
        }
    }

    fn with_filename(f: &str) -> WindowsPath {
        assert ! str::any(f, |c| windows::is_sep(c as u8));
        self.dir_path().push_components(~[str::from_slice(f)])
    }

    fn with_filestem(s: &str) -> WindowsPath {
        match self.filetype() {
          none => self.with_filename(s),
          some(ref t) =>
          self.with_filename(str::from_slice(s) + "." + *t)
        }
    }

    fn with_filetype(t: &str) -> WindowsPath {
        if t.len() == 0 {
            match self.filestem() {
              none => copy self,
              some(s) => self.with_filename(s)
            }
        } else {
            let t = ~"." + str::from_slice(t);
            match self.filestem() {
              none => self.with_filename(t),
              some(ref s) =>
              self.with_filename(*s + t)
            }
        }
    }

    fn dir_path() -> WindowsPath {
        if self.components.len() != 0 {
            self.pop_component()
        } else {
            copy self
        }
    }

    fn file_path() -> WindowsPath {
        let cs = match self.filename() {
          none => ~[],
          some(ref f) => ~[copy *f]
        };
        return WindowsPath { host: none,
                             device: none,
                             is_absolute: false,
                             components: cs }
    }

    fn push_components(cs: &[~str]) -> WindowsPath {
        return WindowsPath { components: normalize(self.components + cs),
                            ..self }
    }

    fn pop_component() -> WindowsPath {
        let mut cs = copy self.components;
        if cs.len() != 0 {
            vec::pop(cs);
        }
        return WindowsPath { components: cs, ..self }
    }
}


fn normalize(components: &[~str]) -> ~[~str] {
    let mut cs = ~[];
    for components.each |c| {
        if c == ~"." { again; }
        if c == ~".." && cs.len() != 0 {
            vec::pop(cs);
            again;
        }
        vec::push(cs, copy c);
    }
    cs
}

mod posix {

    #[test]
    fn test_posix_paths() {
        fn mk(s: &str) -> PosixPath { from_str::<PosixPath>(s) }
        fn t(wp: &PosixPath, s: &str) {
            let ss = wp.to_str();
            let sss = str::from_slice(s);
            if (ss != sss) {
                debug!("got %s", ss);
                debug!("expected %s", sss);
                assert ss == sss;
            }
        }

        t(&(mk("hi")), "hi");
        t(&(mk("hi/there")), "hi/there");
        t(&(mk("hi/there.txt")), "hi/there.txt");

        t(&(mk("hi/there.txt")), "hi/there.txt");
        t(&(mk("hi/there.txt")
           .with_filetype("")), "hi/there");

        t(&(mk("/a/b/c/there.txt")
            .with_dirname("hi")), "hi/there.txt");

        t(&(mk("hi/there.txt")
            .with_dirname(".")), "there.txt");

        t(&(mk("a/b/../c/././/../foo.txt/")),
          "a/foo.txt");

        t(&(mk("a/b/c")
            .push_components([~".."])), "a/b");

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
            .push_components([~"lib", ~"thingy.so"])
            .with_filestem("librustc")),
          "/usr/bin/rust/lib/librustc.so");

    }

}

// Various windows helpers, and tests for the impl.
mod windows {

    #[inline(always)]
    fn is_sep(u: u8) -> bool {
        u == '/' as u8 || u == '\\' as u8
    }

    fn extract_unc_prefix(s: &str) -> option<(~str,~str)> {
        if (s.len() > 1 &&
            s[0] == '\\' as u8 &&
            s[1] == '\\' as u8) {
            let mut i = 2;
            while i < s.len() {
                if s[i] == '\\' as u8 {
                    let pre = s.slice(2, i);
                    let rest = s.slice(i, s.len());
                    return some((pre, rest));
                }
                i += 1;
            }
        }
        none
    }

    fn extract_drive_prefix(s: &str) -> option<(~str,~str)> {
        if (s.len() > 1 &&
            libc::isalpha(s[0] as libc::c_int) != 0 &&
            s[1] == ':' as u8) {
            let rest = if s.len() == 2 { ~"" } else { s.slice(2, s.len()) };
            return some((s.slice(0,1), rest));
        }
        none
    }

    #[test]
    fn test_extract_unc_prefixes() {
        assert extract_unc_prefix("\\\\") == none;
        assert extract_unc_prefix("\\\\hi") == none;
        assert extract_unc_prefix("\\\\hi\\") == some((~"hi", ~"\\"));
        assert extract_unc_prefix("\\\\hi\\there") ==
            some((~"hi", ~"\\there"));
        assert extract_unc_prefix("\\\\hi\\there\\friends.txt") ==
            some((~"hi", ~"\\there\\friends.txt"));
    }

    #[test]
    fn test_extract_drive_prefixes() {
        assert extract_drive_prefix("c") == none;
        assert extract_drive_prefix("c:") == some((~"c", ~""));
        assert extract_drive_prefix("d:") == some((~"d", ~""));
        assert extract_drive_prefix("z:") == some((~"z", ~""));
        assert extract_drive_prefix("c:\\hi") == some((~"c", ~"\\hi"));
        assert extract_drive_prefix("d:hi") == some((~"d", ~"hi"));
        assert extract_drive_prefix("c:hi\\there.txt") ==
            some((~"c", ~"hi\\there.txt"));
        assert extract_drive_prefix("c:\\hi\\there.txt") ==
            some((~"c", ~"\\hi\\there.txt"));
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
            .push_components([~"lib", ~"thingy.dll"])
            .with_filename("librustc.dll")),
          "c:\\program files (x86)\\rust\\lib\\librustc.dll");

    }

}
