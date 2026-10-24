extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn assert_no_attributes(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This will count the "attributes" (in reality the number of hash symbols) on the item.
    assert_eq!(item.to_string().chars().filter(|c| *c == '#').count(), 0);
    item
}

#[proc_macro_attribute]
pub fn assert_one_attribute(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This will count the "attributes" (in reality the number of hash symbols) on the item.
    assert_eq!(item.to_string().chars().filter(|c| *c == '#').count(), 1);
    item
}

#[proc_macro_attribute]
pub fn assert_two_attributes(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This will count the "attributes" (in reality the number of hash symbols) on the item.
    assert_eq!(item.to_string().chars().filter(|c| *c == '#').count(), 2);
    item
}
