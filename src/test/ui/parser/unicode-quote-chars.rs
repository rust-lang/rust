// ignore-tidy-linelength

fn main() {
    println!(“hello world”);
    //~^ ERROR unknown start of token: \u{201c}
    //~^^ HELP Unicode characters '“' (Left Double Quotation Mark) and '”' (Right Double Quotation Mark) look like '"' (Quotation Mark), but are not
}
