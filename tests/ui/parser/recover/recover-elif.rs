fn main() {
    if 1 > 2 {
        println!("Hello.");
   } elif 2 > 1 {
       //~^ expected one of `!`, `.`, `::`, `;`, `?`, `{`, `}`, or an operator, found `2`
        println!("Bye.");
    }
}
