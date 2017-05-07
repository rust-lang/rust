// rustfmt-fn_call_style: Block
// Function call style

fn main() {
    lorem(
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
    );
    // #1501
    let hyper = Arc::new(Client::with_connector(HttpsConnector::new(TlsClient::new())));
}
