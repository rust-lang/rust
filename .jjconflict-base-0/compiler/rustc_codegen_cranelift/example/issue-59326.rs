// Based on https://github.com/rust-lang/rust/blob/689511047a75a30825e367d4fd45c74604d0b15e/tests/ui/issues/issue-59326.rs#L1
// check-pass
trait Service {
    type S;
}

trait Framing {
    type F;
}

impl Framing for () {
    type F = ();
}

trait HttpService<F: Framing>: Service<S = F::F> {}

type BoxService = Box<dyn HttpService<(), S = ()>>;

fn build_server<F: FnOnce() -> BoxService>(_: F) {}

fn make_server<F: Framing>() -> Box<dyn HttpService<F, S = F::F>> {
    unimplemented!()
}

fn main() {
    build_server(|| make_server())
}
