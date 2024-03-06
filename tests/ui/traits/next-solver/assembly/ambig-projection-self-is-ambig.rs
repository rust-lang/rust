//@ check-pass
//@ compile-flags: -Znext-solver

trait Reader: Default {
    fn read_u8_array<A>(&self) -> Result<A, ()> {
        todo!()
    }

    fn read_u8(&self) -> Result<u8, ()> {
        let a: [u8; 1] = self.read_u8_array::<_>()?;
        // This results in a nested `<Result<?0, ()> as Try>::Residual: Sized` goal.
        // The self type normalizes to `?0`. We previously did not force that to be
        // ambiguous but instead incompletely applied the `Self: Sized` candidate
        // from the `ParamEnv`, resulting in a type error.
        Ok(a[0])
    }
}

fn main() {}
