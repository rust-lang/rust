use {
    crate::{TextRange, TextSize},
    serde::{de, Deserialize, Deserializer, Serialize, Serializer},
};

impl Serialize for TextSize {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.raw.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TextSize {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        u32::deserialize(deserializer).map(TextSize::from)
    }
}

impl Serialize for TextRange {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (self.start(), self.end()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TextRange {
    #[allow(clippy::nonminimal_bool)]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (start, end) = Deserialize::deserialize(deserializer)?;
        if !(start <= end) {
            return Err(de::Error::custom(format!(
                "invalid range: {:?}..{:?}",
                start, end
            )));
        }
        Ok(TextRange::new(start, end))
    }
}
