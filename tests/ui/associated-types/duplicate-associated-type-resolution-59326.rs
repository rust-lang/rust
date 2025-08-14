// https://github.com/rust-lang/rust/issues/59326
//@ check-pass
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
