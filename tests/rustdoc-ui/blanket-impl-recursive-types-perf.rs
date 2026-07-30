//@ check-pass
// Regression test: blanket impls on recursive generic types caused
// `cargo doc` to take minutes instead of milliseconds.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub trait Erasable: Sync {}
impl<T> Erasable for T where T: Sync {}

pub trait SessionParameters {
    type Verifier;
}

pub struct Node<T>(pub Arc<T>);

pub struct ComputeScalar<SP: SessionParameters> {
    pub args: BTreeMap<String, ComputeScalarArg<SP>>,
    pub dependencies: Dependency<SP>,
}

pub struct Collect<SP: SessionParameters> {
    pub values: CollectArg<SP>,
    pub dependencies: Dependency<SP>,
}

pub struct ComputeMapping<SP: SessionParameters> {
    pub args: BTreeMap<String, ComputeMappingArg<SP>>,
    pub dependencies: Dependency<SP>,
}

pub struct SendBC<SP: SessionParameters> {
    pub data: Node<SerializeAndSignBC<SP>>,
    pub destinations: BTreeSet<SP::Verifier>,
    pub dependencies: Dependency<SP>,
}

pub struct SerializeAndSignBC<SP: SessionParameters> {
    pub data: Node<ComputeScalar<SP>>,
    pub dependencies: Dependency<SP>,
}

pub struct SerializeAndSignDM<SP: SessionParameters> {
    pub data: DirectMessageArg<SP>,
    pub dependencies: Dependency<SP>,
}

pub struct DeserializeAndCheck<SP: SessionParameters> {
    pub data: Node<Receive<SP>>,
    pub dependencies: Dependency<SP>,
}

pub struct SendDM<SP: SessionParameters> {
    pub data: Node<SerializeAndSignDM<SP>>,
    pub dependencies: Dependency<SP>,
}

pub struct Receive<SP: SessionParameters> {
    pub dependencies: Dependency<SP>,
}

pub struct MergeScalars<SP: SessionParameters> {
    pub left: ComputeScalarArg<SP>,
    pub right: ComputeScalarArg<SP>,
}

pub enum ComputeScalarArg<SP: SessionParameters> {
    ComputeScalar(Node<ComputeScalar<SP>>),
    MergeScalars(Node<MergeScalars<SP>>),
    Collect(Node<Collect<SP>>),
}

pub enum ComputeMappingArg<SP: SessionParameters> {
    ComputeScalar(Node<ComputeScalar<SP>>),
    MergeScalars(Node<MergeScalars<SP>>),
    Collect(Node<Collect<SP>>),
    ComputeMapping(Node<ComputeMapping<SP>>),
    SerializeAndSignBC(Node<SerializeAndSignBC<SP>>),
    SerializeAndSignDM(Node<SerializeAndSignDM<SP>>),
    DeserializeAndCheck(Node<DeserializeAndCheck<SP>>),
}

pub enum CollectArg<SP: SessionParameters> {
    ComputeMapping(Node<ComputeMapping<SP>>),
    SerializeAndSign(Node<SerializeAndSignDM<SP>>),
    DeserializeAndCheck(Node<DeserializeAndCheck<SP>>),
    Send(Node<SendDM<SP>>),
    Receive(Node<Receive<SP>>),
}

pub enum DirectMessageArg<SP: SessionParameters> {
    ComputeScalar(Node<ComputeScalar<SP>>),
    ComputeMapping(Node<ComputeMapping<SP>>),
    DeserializeAndCheck(Node<DeserializeAndCheck<SP>>),
}

pub enum Dependency<SP: SessionParameters> {
    ComputeScalar(Node<ComputeScalar<SP>>),
    Collect(Node<Collect<SP>>),
    MergeScalars(Node<MergeScalars<SP>>),
    SendBC(Node<SendBC<SP>>),
}
