#![deny(disjoint_capture_drop_reorder)]
#![feature(rustc_attrs)]

#[rustc_insignificant_dtor]
struct InsignificantDropPoint {
    x: i32,
    y: i32,
}

impl Drop for InsignificantDropPoint {
    fn drop(&mut self) {}
}

struct Person {
    age: i32,
    name: String,
}

#[rustc_insignificant_dtor]
struct TwoPeople {
    p1: Person,
    p2: Person,
}


// `t` needs Drop because one of its elements needs drop,
// therefore precise capture might affect drop ordering
fn test1_all_need_migration() {
    let t = (InsignificantDropPoint{x: 4, y: 9}, InsignificantDropPoint{x: 4, y: 9});
    let t1 = (InsignificantDropPoint{x: 4, y: 9}, InsignificantDropPoint{x: 4, y: 9});
    let t2 = (InsignificantDropPoint{x: 4, y: 9}, InsignificantDropPoint{x: 4, y: 9});

    let c = || {
    //~^ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| NOTE: drop(&(t, t1, t2));
        let _t = t.0;
        let _t1 = t1.0;
        let _t2 = t2.0;
    };

    c();
}

fn test2_nested_drop_type() {

    let p1 = Person{age:0, name: "foo".to_string()};
    let p2 = Person{age:100, name: "bar".to_string()};
    let two_people = TwoPeople{p1, p2};

    let c = || {
    //~^ERROR: drop order affected for closure because of `capture_disjoint_fields`
    //~| NOTE: drop(&two_people);
        let _t = two_people.p1;
    };

    c();


}

fn main() {
    test1_all_need_migration();
    test2_nested_drop_type();
}