use SyntaxKind;
use syntax_kinds::ERROR;

pub(super) mod imp;
use self::imp::ParserImpl;

pub(crate) struct Parser<'t>(pub(super) ParserImpl<'t>);


impl<'t> Parser<'t> {
    pub(crate) fn current(&self) -> SyntaxKind {
        self.nth(0)
    }

    pub(crate) fn nth(&self, n: u32) -> SyntaxKind {
        self.0.nth(n)
    }

    pub(crate) fn at(&self, kind: SyntaxKind) -> bool {
        self.current() == kind
    }

    pub(crate) fn at_kw(&self, t: &str) -> bool {
        self.0.at_kw(t)
    }

    pub(crate) fn start(&mut self) -> Marker {
        Marker(self.0.start())
    }

    pub(crate) fn bump(&mut self) {
        self.0.bump();
    }

    pub(crate) fn bump_remap(&mut self, kind: SyntaxKind) {
        self.0.bump_remap(kind);
    }

    pub(crate) fn error<T: Into<String>>(&mut self, message: T) {
        self.0.error(message.into())
    }

    pub(crate) fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.at(kind) {
            self.bump();
            return true;
        }
        self.error(format!("expected {:?}", kind));
        false
    }

    pub(crate) fn eat(&mut self, kind: SyntaxKind) -> bool {
        if !self.at(kind) {
            return false;
        }
        self.bump();
        true
    }

    pub(crate) fn err_and_bump(&mut self, message: &str) {
        let m = self.start();
        self.error(message);
        self.bump();
        m.complete(self, ERROR);
    }
}

pub(crate) struct Marker(u32);

impl Marker {
    pub(crate) fn complete(self, p: &mut Parser, kind: SyntaxKind) -> CompletedMarker {
        let pos = self.0;
        ::std::mem::forget(self);
        p.0.complete(pos, kind);
        CompletedMarker(pos)
    }

    pub(crate) fn abandon(self, p: &mut Parser) {
        let pos = self.0;
        ::std::mem::forget(self);
        p.0.abandon(pos);
    }
}

impl Drop for Marker {
    fn drop(&mut self) {
        if !::std::thread::panicking() {
            panic!("Marker must be either completed or abandoned");
        }
    }
}


pub(crate) struct CompletedMarker(u32);

impl CompletedMarker {
    pub(crate) fn precede(self, p: &mut Parser) -> Marker {
        Marker(p.0.precede(self.0))
    }
}
