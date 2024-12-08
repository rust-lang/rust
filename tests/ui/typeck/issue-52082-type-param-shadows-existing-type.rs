// Fix issue 52082: Confusing error if accidentally defining a type parameter with the same name as
// an existing type
//
// To this end, make sure that when trying to retrieve a field of a (reference to) type parameter,
// rustc points to the point where the parameter was defined.
#[derive(Debug)]
struct Point
{
    x: i32,
    y: i32
}

impl Point
{
    fn add(a: &Point, b: &Point) -> Point
    {
        Point {x: a.x + b.x, y: a.y + b.y}
    }
}

trait Eq
{
    fn equals_ref<T>(a: &T, b: &T) -> bool;
    fn equals_val<T>(a: T, b: T) -> bool;
}

impl Eq for Point
{
    fn equals_ref<Point>(a: &Point, b: &Point) -> bool
    {
        a.x == b.x && a.y == b.y //~ ERROR no field `x` on type `&Point` [E0609]
                                 //~|ERROR no field `x` on type `&Point` [E0609]
                                 //~|ERROR no field `y` on type `&Point` [E0609]
                                 //~|ERROR no field `y` on type `&Point` [E0609]
    }

    fn equals_val<Point>(a: Point, b: Point) -> bool
    {
        a.x == b.x && a.y == b.y //~ ERROR no field `x` on type `Point` [E0609]
                                 //~|ERROR no field `x` on type `Point` [E0609]
                                 //~|ERROR no field `y` on type `Point` [E0609]
                                 //~|ERROR no field `y` on type `Point` [E0609]
    }
}

fn main()
{
    let p1 = Point {x:  0, y: 10};
    let p2 = Point {x: 20, y: 42};
    println!("{:?}", Point::add(&p1, &p2));
    println!("p1: {:?}, p2: {:?}", p1, p2);
    println!("&p1 == &p2: {:?}", Point::equals_ref(&p1, &p2));
    println!("p1 == p2: {:?}", Point::equals_val(p1, p2));
}
