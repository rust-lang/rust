// compile-flags: -Z parse-only

struct s {
    bar: ();
    //~^ ERROR expected `,`, or `}`, found `;`
}
