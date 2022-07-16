fn main() {
    // JavaScript-style arrow function.
    let _lambda = (a, b) => { println!(); a + b };
//~^ ERROR expected one of `.`, `;`, `?`, `else`, or an operator, found `=>`
}
