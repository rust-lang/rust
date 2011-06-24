// xfail-stage0

fn main() {
    // We should be able to type infer inside of lambdas.
    auto f = fn () { auto i = 10; };
}
