


// -*- rust -*-

// rustboot can't transmit nils across channels because they don't have
// any size, but rustc currently can because they do have size. Whether
// or not this is desirable I don't know, but here's a regression test.
fn main() {
    let port[()] po = port();
    let chan[()] ch = chan(po);
    ch <| ();
    let () n;
    po |> n;
    assert (n == ());
}