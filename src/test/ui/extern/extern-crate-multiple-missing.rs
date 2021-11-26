// If multiple `extern crate` resolutions fail each of them should produce an error
extern crate bar; //~ ERROR can't find crate for `bar`
extern crate foo; //~ ERROR can't find crate for `foo`

fn main() {
    foo::something();
    bar::something();
}
