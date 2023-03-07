struct NonCopy;

fn main() {
    let array = [NonCopy; 1];
    let _value = array[0];  //~ ERROR [E0508]
}
