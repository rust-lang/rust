// FIXME(fmease): Fully or partially consolidate w/ test `tests/ui/parser/invalid-tuple-indices.rs`.

struct S(u8, (u8, u8));

fn main() {
    let s = S(0, (0, 0));

    { s.1e1; } //~ ERROR invalid tuple index
    //~^ ERROR no field `1e1` on type `S`

    { s.1.; } //~ ERROR unexpected token: `;`

    { s.1.1; }

    { s.1.1e1; } //~ ERROR invalid tuple index
    //~^ ERROR no field `1e1` on type `(u8, u8)`

    { s.1e+; } //~ ERROR unexpected token: `1e+`
               //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1e+`
               //~| ERROR expected at least one digit in exponent

    { s.1e-; } //~ ERROR unexpected token: `1e-`
               //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1e-`
               //~| ERROR expected at least one digit in exponent

    { s.1e+1; } //~ ERROR unexpected token: `1e+1`
                //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1e+1`

    { s.1e-1; } //~ ERROR unexpected token: `1e-1`
                //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1e-1`

    { s.1.1e+1; } //~ ERROR unexpected token: `1.1e+1`
                  //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1.1e+1`

    { s.1.1e-1; } //~ ERROR unexpected token: `1.1e-1`
                  //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1.1e-1`

    { s.0x1e1; } //~ ERROR invalid tuple index
    //~^ ERROR no field `11` on type `S`

    { s.0x1.; } //~ ERROR hexadecimal float literal is not supported
                //~| ERROR unexpected token: `0x1.`
                //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `0x1.`

    { s.0x1.1; } //~ ERROR hexadecimal float literal is not supported
                 //~| ERROR unexpected token: `0x1.1`
                 //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `0x1.1`

    { s.0x1.1e1; } //~ ERROR hexadecimal float literal is not supported
                   //~| ERROR unexpected token: `0x1.1e1`
                   //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `0x1.1e1`

    { s.0x1e+; } //~ ERROR invalid tuple index
    //~^ ERROR expected expression, found `;`

    { s.0x1e-; } //~ ERROR invalid tuple index
    //~^ ERROR expected expression, found `;`

    { s.0x1e+1; } //~ ERROR invalid tuple index
    //~^ ERROR cannot add `{integer}` to `(u8, u8)`

    { s.0x1e-1; } //~ ERROR invalid tuple index
    //~^ ERROR cannot subtract `{integer}` from `(u8, u8)`

    { s.0x1.1e+1; } //~ ERROR unexpected token: `0x1.1e+1`
                    //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `0x1.1e+1`
                    //~| ERROR hexadecimal float literal is not supported

    { s.0x1.1e-1; } //~ ERROR unexpected token: `0x1.1e-1`
                    //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `0x1.1e-1`
                    //~| ERROR hexadecimal float literal is not supported

    { s.1e1f32; } //~ ERROR invalid tuple index
    //~^ ERROR no field `1e1` on type `S`
    //~| ERROR suffixes on a tuple index are invalid

    { s.1.f32; } //~ ERROR no field `f32` on type `(u8, u8)`

    { s.1.1f32; } //~ ERROR suffixes on a tuple index are invalid

    { s.1.1e1f32; } //~ ERROR invalid tuple index
    //~^ ERROR suffixes on a tuple index are invalid
    //~| ERROR no field `1e1` on type `(u8, u8)`

    { s.1e+f32; } //~ ERROR unexpected token: `1e+f32`
                  //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1e+f32`
                  //~| ERROR expected at least one digit in exponent

    { s.1e-f32; } //~ ERROR unexpected token: `1e-f32`
                  //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1e-f32`
                  //~| ERROR expected at least one digit in exponent

    { s.1e+1f32; } //~ ERROR unexpected token: `1e+1f32`
                   //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1e+1f32`

    { s.1e-1f32; } //~ ERROR unexpected token: `1e-1f32`
                   //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1e-1f32`

    { s.1.1e+1f32; } //~ ERROR unexpected token: `1.1e+1f32`
                    //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1.1e+1f32`

    { s.1.1e-1f32; } //~ ERROR unexpected token: `1.1e-1f32`
                    //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `1.1e-1f32`
}
