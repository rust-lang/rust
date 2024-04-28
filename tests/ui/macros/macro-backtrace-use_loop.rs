///```
/// ///use_loop macro can create a loop
/// ///this macro take Five values
/// /// 1 - if you want to do event at loop set True else set False
/// /// 2 - start-Number
/// /// 3 - End-Number
/// /// 4 - variable for loop
/// /// 5 - the method
/// /// you should type true at first value to method working
/// /// example
/// use_loop!(true, 0, 100, i, akp!("{}", i));
/// /// dont do this
///  use_loop!(false, 0, 100, i, akp!("{}", i)); /// Syntax Error
/// ```
#[macro_export]
macro_rules! use_loop {
    ($should_execute:expr, $start_number:expr, $end_number:expr, $theVar:ident, $the_method:expr) => {
        for $theVar in $start_number..$end_number {
            if $should_execute {
                $the_method;
            }
        }
    };
}

fn main() {
    use_loop!(true, 0, 100, i, akp!("{}", i));
}
