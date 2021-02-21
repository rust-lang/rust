#![feature(capture_disjoint_fields)]
#![feature(rustc_attrs)]

fn main() {
    let _z = 9;
    let t = (String::from("Hello"), String::from("World"));
    let g = (String::from("Mr"), String::from("Goose"));

    let a = #[rustc_capture_analysis] || {
        let (_, g2) = g;
        println!("{}", g2);
        let c = #[rustc_capture_analysis] ||  {
            let (_, t2) = t;
            println!("{}", t2);
        };

        c();
    };

    a();
}
