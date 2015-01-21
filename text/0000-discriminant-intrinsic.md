- Start Date: 2015-01-21
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add a new intrinsic, `discriminant_value` that extracts the value of the discriminant for enum
types.

# Motivation

Many operations that work with discriminant values can be significantly improved with the ability to
extract the value of the discriminant that is used to distinguish between variants in an enum. While
trivial cases often optimise well, more complex ones would benefit from direct access to this value.

A good example is the `SqlState` enum from the `postgres` crate (Listed at the end of this RFC). It
contains 233 variants, of which all but one contain no fields. The most obvious implementation of
(for example) the `PartialEq` trait looks like this:

```rust
match (self, other) {
    (&Unknown(ref s1), &Unknown(ref s2)) => s1 == s2,
    (&SuccessfulCompletion, &SuccessfulCompletion) => true,
    (&Warning, &Warning) => true,
    (&DynamicResultSetsReturned, &DynamicResultSetsReturned) => true,
    (&ImplicitZeroBitPadding, &ImplicitZeroBitPadding) => true,
	      .
		  .
		  .
    (_, _) => false
}
```

Even with optimisations enabled, this code is very suboptimal, producing
[this code](https://gist.github.com/Aatch/c23a45634b10aaecad05). A way to extract the discriminant
would allow this code:

```rust
match (self, other) {
    (&Unknown(ref s1), &Unknown(ref s2)) => s1 == s2,
    (l, r) => unsafe {
	    discriminant_value(l) == discriminant(r)
	}
}
```

Which is compiled into [this IR](https://gist.github.com/Aatch/beb736b93a908aa67e84).

# Detailed design

## What is a discriminant?

A discriminant is a value stored in an enum type that indicates which variant the value is. The most
common case is that the discriminant is stored directly as an extra field in the variant. However,
the discriminant may be stored in any place, and in any format. However, we can always extract the
discriminant from the value somehow.

## Implementation

For any given type, `discriminant_value` will return a `u64` value. The values returned are as
specified:

* **Non-Enum Type**: Always 0
* **Enum Type**: A value that uniquely identifies that variant within its type. I.E. for a given
  enum typem `E`, and two values of type `E`, `a` and `b`, the expression `discriminant_value(a) ==
  discriminant_value(b)` is true iff `a` and `b` are the same variant. Two values of different types
  may return the same discriminant value.

The reason for this specification is to allow flexibilty in usage of the intrinsic without
compromising our ability to change the representation at will.

# Drawbacks

* Potentially exposes implementation details. However, relying the specific values returned from
`discriminant_value` should be considered bad practice, as the intrinsic provides no such guarantee.

* Does not allow for the value to be used as part of ordering.

* Allows non-enum types to be provided. This may be unexpected by some users.

# Alternatives

* More strongly specify the values returned. This would allow for a broader range of uses, but
  requires specifying behaviour that we may not want to.

* Disallow non-enum types. Non-enum types do not have a discriminant, so trying to extract might be
  considered an error. However, there is no compelling reason to disallow these types as we can
  simply treat them as single-variant enums and synthesise a zero constant. Note that this is what
  would be done for single-variant enums anyway.

* Do nothing. Improvements to codegen and/or optimisation could make this uneccessary. The
  "Sufficiently Smart Compiler" trap is a strong case against this reasoning though. There will
  likely always be cases where the user can write more efficient code than the compiler can produce.

# Unresolved questions

* Should `#[derive]` use this intrinsic to improve derived implementations of traits? While
  intrinsics are inherently unstable, `#[derive]`d code is compiler generated and therefore can be
  updated if the intrinsic is changed or removed.

# Appendix

```rust
pub enum SqlState {
    SuccessfulCompletion,
    Warning,
    DynamicResultSetsReturned,
    ImplicitZeroBitPadding,
    NullValueEliminatedInSetFunction,
    PrivilegeNotGranted,
    PrivilegeNotRevoked,
    StringDataRightTruncationWarning,
    DeprecatedFeature,
    NoData,
    NoAdditionalDynamicResultSetsReturned,
    SqlStatementNotYetComplete,
    ConnectionException,
    ConnectionDoesNotExist,
    ConnectionFailure,
    SqlclientUnableToEstablishSqlconnection,
    SqlserverRejectedEstablishmentOfSqlconnection,
    TransactionResolutionUnknown,
    ProtocolViolation,
    TriggeredActionException,
    FeatureNotSupported,
    InvalidTransactionInitiation,
    LocatorException,
    InvalidLocatorException,
    InvalidGrantor,
    InvalidGrantOperation,
    InvalidRoleSpecification,
    DiagnosticsException,
    StackedDiagnosticsAccessedWithoutActiveHandler,
    CaseNotFound,
    CardinalityViolation,
    DataException,
    ArraySubscriptError,
    CharacterNotInRepertoire,
    DatetimeFieldOverflow,
    DivisionByZero,
    ErrorInAssignment,
    EscapeCharacterConflict,
    IndicatorOverflow,
    IntervalFieldOverflow,
    InvalidArgumentForLogarithm,
    InvalidArgumentForNtileFunction,
    InvalidArgumentForNthValueFunction,
    InvalidArgumentForPowerFunction,
    InvalidArgumentForWidthBucketFunction,
    InvalidCharacterValueForCast,
    InvalidDatetimeFormat,
    InvalidEscapeCharacter,
    InvalidEscapeOctet,
    InvalidEscapeSequence,
    NonstandardUseOfEscapeCharacter,
    InvalidIndicatorParameterValue,
    InvalidParameterValue,
    InvalidRegularExpression,
    InvalidRowCountInLimitClause,
    InvalidRowCountInResultOffsetClause,
    InvalidTimeZoneDisplacementValue,
    InvalidUseOfEscapeCharacter,
    MostSpecificTypeMismatch,
    NullValueNotAllowedData,
    NullValueNoIndicatorParameter,
    NumericValueOutOfRange,
    StringDataLengthMismatch,
    StringDataRightTruncationException,
    SubstringError,
    TrimError,
    UnterminatedCString,
    ZeroLengthCharacterString,
    FloatingPointException,
    InvalidTextRepresentation,
    InvalidBinaryRepresentation,
    BadCopyFileFormat,
    UntranslatableCharacter,
    NotAnXmlDocument,
    InvalidXmlDocument,
    InvalidXmlContent,
    InvalidXmlComment,
    InvalidXmlProcessingInstruction,
    IntegrityConstraintViolation,
    RestrictViolation,
    NotNullViolation,
    ForeignKeyViolation,
    UniqueViolation,
    CheckViolation,
    ExclusionViolation,
    InvalidCursorState,
    InvalidTransactionState,
    ActiveSqlTransaction,
    BranchTransactionAlreadyActive,
    HeldCursorRequiresSameIsolationLevel,
    InappropriateAccessModeForBranchTransaction,
    InappropriateIsolationLevelForBranchTransaction,
    NoActiveSqlTransactionForBranchTransaction,
    ReadOnlySqlTransaction,
    SchemaAndDataStatementMixingNotSupported,
    NoActiveSqlTransaction,
    InFailedSqlTransaction,
    InvalidSqlStatementName,
    TriggeredDataChangeViolation,
    InvalidAuthorizationSpecification,
    InvalidPassword,
    DependentPrivilegeDescriptorsStillExist,
    DependentObjectsStillExist,
    InvalidTransactionTermination,
    SqlRoutineException,
    FunctionExecutedNoReturnStatement,
    ModifyingSqlDataNotPermittedSqlRoutine,
    ProhibitedSqlStatementAttemptedSqlRoutine,
    ReadingSqlDataNotPermittedSqlRoutine,
    InvalidCursorName,
    ExternalRoutineException,
    ContainingSqlNotPermitted,
    ModifyingSqlDataNotPermittedExternalRoutine,
    ProhibitedSqlStatementAttemptedExternalRoutine,
    ReadingSqlDataNotPermittedExternalRoutine,
    ExternalRoutineInvocationException,
    InvalidSqlstateReturned,
    NullValueNotAllowedExternalRoutine,
    TriggerProtocolViolated,
    SrfProtocolViolated,
    SavepointException,
    InvalidSavepointException,
    InvalidCatalogName,
    InvalidSchemaName,
    TransactionRollback,
    TransactionIntegrityConstraintViolation,
    SerializationFailure,
    StatementCompletionUnknown,
    DeadlockDetected,
    SyntaxErrorOrAccessRuleViolation,
    SyntaxError,
    InsufficientPrivilege,
    CannotCoerce,
    GroupingError,
    WindowingError,
    InvalidRecursion,
    InvalidForeignKey,
    InvalidName,
    NameTooLong,
    ReservedName,
    DatatypeMismatch,
    IndeterminateDatatype,
    CollationMismatch,
    IndeterminateCollation,
    WrongObjectType,
    UndefinedColumn,
    UndefinedFunction,
    UndefinedTable,
    UndefinedParameter,
    UndefinedObject,
    DuplicateColumn,
    DuplicateCursor,
    DuplicateDatabase,
    DuplicateFunction,
    DuplicatePreparedStatement,
    DuplicateSchema,
    DuplicateTable,
    DuplicateAliaas,
    DuplicateObject,
    AmbiguousColumn,
    AmbiguousFunction,
    AmbiguousParameter,
    AmbiguousAlias,
    InvalidColumnReference,
    InvalidColumnDefinition,
    InvalidCursorDefinition,
    InvalidDatabaseDefinition,
    InvalidFunctionDefinition,
    InvalidPreparedStatementDefinition,
    InvalidSchemaDefinition,
    InvalidTableDefinition,
    InvalidObjectDefinition,
    WithCheckOptionViolation,
    InsufficientResources,
    DiskFull,
    OutOfMemory,
    TooManyConnections,
    ConfigurationLimitExceeded,
    ProgramLimitExceeded,
    StatementTooComplex,
    TooManyColumns,
    TooManyArguments,
    ObjectNotInPrerequisiteState,
    ObjectInUse,
    CantChangeRuntimeParam,
    LockNotAvailable,
    OperatorIntervention,
    QueryCanceled,
    AdminShutdown,
    CrashShutdown,
    CannotConnectNow,
    DatabaseDropped,
    SystemError,
    IoError,
    UndefinedFile,
    DuplicateFile,
    ConfigFileError,
    LockFileExists,
    FdwError,
    FdwColumnNameNotFound,
    FdwDynamicParameterValueNeeded,
    FdwFunctionSequenceError,
    FdwInconsistentDescriptorInformation,
    FdwInvalidAttributeValue,
    FdwInvalidColumnName,
    FdwInvalidColumnNumber,
    FdwInvalidDataType,
    FdwInvalidDataTypeDescriptors,
    FdwInvalidDescriptorFieldIdentifier,
    FdwInvalidHandle,
    FdwInvalidOptionIndex,
    FdwInvalidOptionName,
    FdwInvalidStringLengthOrBufferLength,
    FdwInvalidStringFormat,
    FdwInvalidUseOfNullPointer,
    FdwTooManyHandles,
    FdwOutOfMemory,
    FdwNoSchemas,
    FdwOptionNameNotFound,
    FdwReplyHandle,
    FdwSchemaNotFound,
    FdwTableNotFound,
    FdwUnableToCreateExcecution,
    FdwUnableToCreateReply,
    FdwUnableToEstablishConnection,
    PlpgsqlError,
    RaiseException,
    NoDataFound,
    TooManyRows,
    InternalError,
    DataCorrupted,
    IndexCorrupted,
    Unknown(String),
}
```