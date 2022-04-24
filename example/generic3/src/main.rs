// 定义范型函数参数为T任何类型都可以
trait Add<T = Self> {
    // 返回的类型x
    type X;
    // 定义方法参数为T类型的任何类型参数，返回值是x类型参数
    fn add(self,t:T) -> Self::X;   
}
struct Point{
    a:i32,
    b:i32,
}
impl Add for Point{
    type X = Point;
    fn add(self,other:Point)->Point{
        Point{
            a:self.a + other.a,
            b:self.b + other.b,
        }
    }
}
fn main() {
    println!("{:?}",Point(a:3,b:3)+Point(a:6,b:8));
}
