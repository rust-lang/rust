// exec-env:RUST_POISON_ON_FREE=1

fn main() {
    let x: @mut @Option<~int> = @mut @None;
    match x {
      @@Some(y) => {
        // here, the refcount of `*x` is bumped so
        // `y` remains valid even if `*x` is modified.
        *x = @None;
      }
      @@None => {
        // here, no bump of the ref count of `*x` is needed, but in
        // fact a bump occurs anyway because of how pattern marching
        // works.
      }
    }
}