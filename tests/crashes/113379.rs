//@ known-bug: #113379

async fn f999() -> Vec<usize> {
    'b: {
        continue 'b;
    }
}
