// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

struct NonCopy;

fn main() {
    let array = [NonCopy; 1];
    let _value = array[0];  //[ast]~ ERROR [E0508]
                            //[mir]~^ ERROR [E0508]
}
