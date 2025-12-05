// Test that a structure which tries to store a pointer to `y` into
// `p` (indirectly) fails to compile.

struct SomeStruct<'a, 'b: 'a> {
    p: &'a mut &'b i32,
    y: &'b i32,
}

fn test() {
    let x = 44;
    let mut p = &x;

    {
        let y = 22;

        let closure = SomeStruct {
            p: &mut p,
            y: &y,
            //~^ ERROR `y` does not live long enough [E0597]
        };

        closure.invoke();
    }

    deref(p);
}

impl<'a, 'b> SomeStruct<'a, 'b> {
    fn invoke(self) {
        *self.p = self.y;
    }
}

fn deref(_: &i32) { }

fn main() { }
