// rustfmt-style_edition: 2027
// rustfmt-max_width: 100

impl<T: CapnpWrite>
    IntoMessage<
        capnp::message::TypedReader<
            capnp::message::Builder<capnp::message::HeapAllocator>,
            T::Capnp,
        >,
    > for T
{
    fn into_msg(
        self,
    ) -> capnp::message::TypedReader<
        capnp::message::Builder<capnp::message::HeapAllocator>,
        T::Capnp,
    > {
        todo!()
    }
}

impl<T> Foo for T {
    fn into_msg_return_type_single_line_99(
        self,
    ) -> some::long::path::to::GenericType<long::path::to::GenericType<some::Type>, some::Type____>
    {
        todo!()
    }

    fn into_msg_return_type_single_line_100(
        self,
    ) -> some::long::path::to::GenericType<long::path::to::GenericType<some::Type>, some::Type_____>
    {
        todo!()
    }

    fn into_msg_return_type_single_line_101(
        self,
    ) -> some::long::path::to::GenericType<
        long::path::to::GenericType<some::Type>,
        some::Type______,
    > {
        todo!()
    }
}
