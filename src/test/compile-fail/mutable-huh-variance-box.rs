// error-pattern: mismatched types

fn main() {
    let v = @mut [0]/~;

    fn f(&&v: @mut [const int]/~) {
        *v = [mut 3]/~
    }

    f(v);
}
