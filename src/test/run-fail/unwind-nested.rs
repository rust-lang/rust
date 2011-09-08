// error-pattern:fail

fn main() {
    let a = @0;
    {
        let b = @0;
        {
            fail;
        }
    }
}