//@ needs-target-std
#[path = "../remap/scrape.rs"]
mod scrape;

fn main() {
    scrape::scrape(&["--scrape-tests"], &[]);
}
