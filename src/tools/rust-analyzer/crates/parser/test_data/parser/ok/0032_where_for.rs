fn test_serialization<SER>()
where
    SER: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
{}
