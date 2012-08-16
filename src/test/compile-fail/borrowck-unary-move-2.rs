struct noncopyable {
    i: (); new() { self.i = (); } drop { #error["dropped"]; }
}
enum wrapper = noncopyable;

fn main() {
    let x1 = wrapper(noncopyable());
    let _x2 = move *x1; //~ ERROR moving out of enum content
}