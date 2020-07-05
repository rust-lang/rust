use std::marker::PhantomData;

pub fn missing_field<'de, V, E>() -> Result<V, E> {
    #[allow(dead_code)]
    struct MissingFieldDeserializer<E>(PhantomData<E>);

    impl<E> Deserializer for MissingFieldDeserializer<E> {}
    unimplemented!()
}

pub trait Deserializer {}
