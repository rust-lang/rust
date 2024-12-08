struct Point(i32, i32);

fn main() {
    let origin = Point(0, 0);
    origin.0;
    origin.1;
    origin.2;
    //~^ ERROR no field `2` on type `Point`
    let tuple = (0, 0);
    tuple.0;
    tuple.1;
    tuple.2;
    //~^ ERROR no field `2` on type `({integer}, {integer})`
}
