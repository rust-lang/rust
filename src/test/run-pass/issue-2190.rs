// xfail-test
type t = {
    f: fn~()
};

fn main() {
    let _t: t = { f: {||()} };
}
