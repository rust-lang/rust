#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

struct Point {
    x: f32,
    y: f32,
}

struct Pentagon {
    points: [Point; 5],
}

fn main() {
    let p1 = Point { x: 10.0, y: 10.0 };
    let p2 = Point { x: 7.5, y: 12.5 };
    let p3 = Point { x: 15.0, y: 15.0 };
    let p4 = Point { x: 12.5, y: 12.5 };
    let p5 = Point { x: 20.0, y: 10.0 };

    let pent = Pentagon { points: [p1, p2, p3, p4, p5] };

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    || {
        println!("{}", pent.points[5].x);
        //~^ ERROR: Capturing pent[(0, 0)] -> ImmBorrow
        //~^^ ERROR: Min Capture pent[(0, 0)] -> ImmBorrow
    };
}
