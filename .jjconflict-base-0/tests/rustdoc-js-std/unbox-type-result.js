// exact-check

// Test case for https://github.com/rust-lang/rust/issues/139665
// make sure that std::io::Result and std::thread::Result get unboxed

const EXPECTED = [
    {
        query: "File -> Metadata",
        others: [
            { path: "std::fs::File", name: "metadata" },
        ]
    },
    {
        query: "JoinHandle<T> -> T",
        others: [
            { path: "std::thread::JoinHandle", name: "join" },
        ]
    },
];
