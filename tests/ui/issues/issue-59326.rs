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
//~^ ERROR associated type bound for `S` in `dyn HttpService` differs from associated type bound from supertrait

fn build_server<F: FnOnce() -> BoxService>(_: F) {}

fn make_server<F: Framing>() -> Box<dyn HttpService<F, S = F::F>> {
    //~^ WARN associated type bound for `S` in `dyn HttpService` is redundant
    unimplemented!()
}

fn main() {
    build_server(|| make_server())
}
