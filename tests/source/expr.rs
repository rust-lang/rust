// Test expressions

fn foo() -> bool {
    let very_long_variable_name = ( a +  first +   simple + test   );
    let very_long_variable_name = (a + first + simple + test + AAAAAAAAAAAAA + BBBBBBBBBBBBBBBBB + b + c);

    let some_val = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa * bbbb / (bbbbbb -
        function_call(x, *very_long_pointer, y))
    + 1000;

some_ridiculously_loooooooooooooooooooooong_function(10000 * 30000000000 + 40000 / 1002200000000
                                                     - 50000 * sqrt(-1),
                                                     trivial_value);
    (((((((((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + a +
             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaa)))))))));

     if  1  + 2 > 0  { let result = 5; result } else { 4};

    if cond() {
        something();
    } else  if different_cond() {
        something_else();
    } else {
        // Check subformatting
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    }
}
