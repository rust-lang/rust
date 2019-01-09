//compile-flags: -Z borrowck=mir

fn foo(_:String) {}

fn main()
{
    let my_str = "hello".to_owned();
    match Some(42) {
        Some(_) if { drop(my_str); false } => {}
        Some(_) => {}
        None => { foo(my_str); } //~ ERROR [E0382]
    }
}
