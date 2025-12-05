// When the type of a method call's receiver is unknown, the span should point
// to the receiver (and not the entire call, as was previously the case before
// the fix of which this tests).

fn shines_a_beacon_through_the_darkness() {
    let x: Option<_> = None; //~ ERROR type annotations needed
    x.unwrap().method_that_could_exist_on_some_type();
}

fn courier_to_des_moines_and_points_west(data: &[u32]) -> String {
    data.iter()
        .sum::<_>() //~ ERROR type annotations needed
        .to_string()
}

fn main() {}
