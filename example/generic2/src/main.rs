// trait范型参数的使用
//定义两个参数类型RHS和Ouput。代表的一个做加法一个是返回值。
trait Add<RHS,Ouput> {
    //type Ouput;
    // 函数里的类型RHS做加法，Ouput是返回类型
    fn add(self,rhs:RHS) ->Ouput;

}
// 实现范型，两个参数类型可以为任意一种，后一种是返回类型
impl Add<i32,i32> for i32{
    fn add(self,rhs:i32) -> i32{
        self + rhs
    }
}
impl Add<u32,i32> for u32 {
    fn add(self,rhs:u32) ->i32{
        (self +rhs) as i32
    }

}
fn main(){
    let (a,b,c,d) = (1i32,2i32,3u32,4u32);
    let x:i32 = a.add(b);
    let y:i32 = c.add(d);
    assert_eq!(x,3i32);
    assert_eq!(y,7i32)

}