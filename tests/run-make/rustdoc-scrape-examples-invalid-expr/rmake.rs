//@ needs-target-std
#[path = "../rustdoc-scrape-examples-remap/scrape.rs"]
mod scrape;

fn main() {
    scrape::scrape(&[]);
}
