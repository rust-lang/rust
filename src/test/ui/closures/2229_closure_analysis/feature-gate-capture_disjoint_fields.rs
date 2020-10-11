#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

fn main() {
    let s = format!("s");

    let c = #[rustc_capture_analysis] || {
    //~^ ERROR: attributes on expressions are experimental
        println!("This uses new capture analyysis to capture s={}", s);
    };
}
