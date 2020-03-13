use {
    crate::{TextRange, TextSize},
    serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer},
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
        Deserialize::deserialize(deserializer).map(TextSize)
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
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (start, end) = Deserialize::deserialize(deserializer)?;
        if !(start <= end) {
            return Err(Error::custom(format!(
                "invalid range: {:?}..{:?}",
                start, end
            )));
        }
        Ok(TextRange(start, end))
    }
}
