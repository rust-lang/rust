// use std :: ops;
// 范型继承
trait Page { 
    fn set_page(&self,p:i32){
        println!("Page Default1");
    }
}
trait PerPage{ 
    fn set_perpage(&self,num:i32){
        println!("Per Page default:10");
    }
}
// 添加其功能
trait pageinate:Page+PerPage{
fn set_skip_page(&self,num:i32){
    println!("Skip page:{:?}",num)
}
}
 struct MyPageinate{
     page :i32
 }  
 impl Page for MyPageinate{}
 impl PerPage for MyPageinate{} 
 impl <T:Page + PerPage> pageinate for T{}
fn main() {
    let my_page =MyPageinate(page: 1);
    my_page.set_page(2);
    my_page.set_perpage(8);
    my_page.set_skip_page(12);
}
