//@ run-pass
//@ compile-flags: -Copt-level=3 -Cdebuginfo=2 -Zmir-opt-level=3
//@ edition: 2021

fn main() {
    TranslatorI.visit_pre();
}

impl TranslatorI {
    fn visit_pre(self) {
        Some(())
            .map(|_| self.flags())
            .unwrap_or_else(|| self.flags());
    }
}

struct TranslatorI;

impl TranslatorI {
    fn flags(&self) {}
}
