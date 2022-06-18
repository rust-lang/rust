
fn takes_a_long_time_to_rustfmt() {
    let inner_cte = vec![Node {
        node: Some(node::Node::CommonTableExpr(Box::new(CommonTableExpr {
            ctename: String::from("ranked_by_age_within_key"),
            aliascolnames: vec![],
            ctematerialized: CteMaterialize::Default as i32,
            ctequery: Some(Box::new(Node {
                node: Some(node::Node::SelectStmt(Box::new(SelectStmt {
                    distinct_clause: vec![],
                    into_clause: None,
                    target_list: vec![
                        Node {
                            node: Some(node::Node::ResTarget(Box::new(ResTarget {
                                name: String::from(""),
                                indirection: vec![],
                                val: Some(Box::new(Node {
                                    node: Some(node::Node::ColumnRef(ColumnRef {
                                        fields: vec![Node {
                                            node: Some(node::Node::AStar(AStar{}))
                                        }],
                                        location: 80
                                    }))
                                })),
                                location: 80
                            })))
                        },
                        Node {
                            node: Some(node::Node::ResTarget(Box::new(ResTarget {
                                name: String::from("rank_in_key"),
                                indirection: vec![],
                                val: Some(Box::new(Node {
                                    node: Some(node::Node::FuncCall(Box::new(FuncCall {
                                        funcname: vec![Node {
                                            node: Some(node::Node::String(String2 {
                                                str: String::from("row_number")
                                            }))
                                        }],
                                        args: vec![],
                                        agg_order: vec![],
                                        agg_filter: None,
                                        agg_within_group: false,
                                        agg_star: false,
                                        agg_distinct: false,
                                        func_variadic: false,
                                        over: Some(Box::new(WindowDef {
                                            name: String::from(""),
                                            refname: String::from(""),
                                            partition_clause: vec![
                                                Node {
                                                    node: Some(node::Node::ColumnRef(ColumnRef {
                                                        fields: vec![Node {
                                                            node: Some(node::Node::String(String2 {
                                                                str: String::from("synthetic_key")
                                                            }))
                                                        }], location: 123
                                                    }))
                                                }], order_clause: vec![Node {
                                                    node: Some(node::Node::SortBy(Box::new(SortBy {
                                                        node: Some(Box::new(Node {
                                                            node: Some(node::Node::ColumnRef(ColumnRef {
                                                                fields: vec![Node {
                                                                    node: Some(node::Node::String(String2 {
                                                                        str: String::from("logical_timestamp")
                                                                    }))
                                                                }], location: 156
                                                            }))
                                                        })),
                                                        sortby_dir: SortByDir::SortbyDesc as i32,
                                                        sortby_nulls: SortByNulls::SortbyNullsDefault as i32,
                                                        use_op: vec![],
                                                        location: -1
                                                    })))
                                                }], frame_options: 1058, start_offset: None, end_offset: None, location: 109
                                            })),
                                            location: 91
                                        })))
                                    })),
                                    location: 91
                                })))
                            }],
                            from_clause: vec![Node {
                                node: Some(node::Node::RangeVar(RangeVar {
                                    catalogname: String::from(""), schemaname: String::from("_supertables"), relname: String::from("9999-9999-9999"), inh: true, relpersistence: String::from("p"), alias: None, location: 206
                                }))
                            }],
                            where_clause: Some(Box::new(Node {
                                node: Some(node::Node::AExpr(Box::new(AExpr {
                                    kind: AExprKind::AexprOp as i32,
                                    name: vec![Node {
                                        node: Some(node::Node::String(String2 {
                                            str: String::from("<=")
                                        }))
                                    }],
                                    lexpr: Some(Box::new(Node {
                                        node: Some(node::Node::ColumnRef(ColumnRef {
                                            fields: vec![Node {
                                                node: Some(node::Node::String(String2 {
                                                    str: String::from("logical_timestamp")
                                                }))
                                            }],
                                            location: 250
                                        }))
                                    })),
                                    rexpr: Some(Box::new(Node {
                                        node: Some(node::Node::AConst(Box::new(AConst {
                                            val: Some(Box::new(Node {
                                                node: Some(node::Node::Integer(Integer {
                                                    ival: 9000
                                                }))
                                            })),
                                            location: 271
                                        })))
                                    })),
                                    location: 268
                                })))
                            })),
                            group_clause: vec![],
                            having_clause: None,
                            window_clause: vec![],
                            values_lists: vec![],
                            sort_clause: vec![],
                            limit_offset: None,
                            limit_count: None,
                            limit_option: LimitOption::Default as i32,
                            locking_clause: vec![],
                            with_clause: None,
                            op: SetOperation::SetopNone as i32,
                            all: false,
                            larg: None,
                            rarg: None
                        }))),
            })),
            location: 29,
            cterecursive: false,
            cterefcount: 0,
            ctecolnames: vec![],
            ctecoltypes: vec![],
            ctecoltypmods: vec![],
            ctecolcollations: vec![],
        }))),
    }];
    let outer_cte = vec![Node {
        node: Some(node::Node::CommonTableExpr(Box::new(CommonTableExpr {
            ctename: String::from("table_name"),
            aliascolnames: vec![],
            ctematerialized: CteMaterialize::Default as i32,
            ctequery: Some(Box::new(Node {
                node: Some(node::Node::SelectStmt(Box::new(SelectStmt {
                    distinct_clause: vec![],
                    into_clause: None,
                    target_list: vec![
                        Node {
                            node: Some(node::Node::ResTarget(Box::new(ResTarget {
                                name: String::from("column1"),
                                indirection: vec![],
                                val: Some(Box::new(Node {
                                    node: Some(node::Node::ColumnRef(ColumnRef {
                                        fields: vec![Node {
                                            node: Some(node::Node::String(String2 {
                                                str: String::from("c1"),
                                            })),
                                        }],
                                        location: 301,
                                    })),
                                })),
                                location: 301,
                            }))),
                        },
                        Node {
                            node: Some(node::Node::ResTarget(Box::new(ResTarget {
                                name: String::from("column2"),
                                indirection: vec![],
                                val: Some(Box::new(Node {
                                    node: Some(node::Node::ColumnRef(ColumnRef {
                                        fields: vec![Node {
                                            node: Some(node::Node::String(String2 {
                                                str: String::from("c2"),
                                            })),
                                        }],
                                        location: 324,
                                    })),
                                })),
                                location: 324,
                            }))),
                        },
                    ],
                    from_clause: vec![Node {
                        node: Some(node::Node::RangeVar(RangeVar {
                            catalogname: String::from(""),
                            schemaname: String::from(""),
                            relname: String::from("ranked_by_age_within_key"),
                            inh: true,
                            relpersistence: String::from("p"),
                            alias: None,
                            location: 347,
                        })),
                    }],
                    where_clause: Some(Box::new(Node {
                        node: Some(node::Node::BoolExpr(Box::new(BoolExpr {
                            xpr: None,
                            boolop: BoolExprType::AndExpr as i32,
                            args: vec![
                                Node {
                                    node: Some(node::Node::AExpr(Box::new(AExpr {
                                        kind: AExprKind::AexprOp as i32,
                                        name: vec![Node {
                                            node: Some(node::Node::String(String2 {
                                                str: String::from("="),
                                            })),
                                        }],
                                        lexpr: Some(Box::new(Node {
                                            node: Some(node::Node::ColumnRef(ColumnRef {
                                                fields: vec![Node {
                                                    node: Some(node::Node::String(
                                                        String2 {
                                                            str: String::from("rank_in_key"),
                                                        },
                                                    )),
                                                }],
                                                location: 382,
                                            })),
                                        })),
                                        rexpr: Some(Box::new(Node {
                                            node: Some(node::Node::AConst(Box::new(AConst {
                                                val: Some(Box::new(Node {
                                                    node: Some(node::Node::Integer(
                                                        Integer { ival: 1 },
                                                    )),
                                                })),
                                                location: 396,
                                            }))),
                                        })),
                                        location: 394,
                                    }))),
                                },
                                Node {
                                    node: Some(node::Node::AExpr(Box::new(AExpr {
                                        kind: AExprKind::AexprOp as i32,
                                        name: vec![Node {
                                            node: Some(node::Node::String(String2 {
                                                str: String::from("="),
                                            })),
                                        }],
                                        lexpr: Some(Box::new(Node {
                                            node: Some(node::Node::ColumnRef(ColumnRef {
                                                fields: vec![Node {
                                                    node: Some(node::Node::String(
                                                        String2 {
                                                            str: String::from("is_deleted"),
                                                        },
                                                    )),
                                                }],
                                                location: 402,
                                            })),
                                        })),
                                        rexpr: Some(Box::new(Node {
                                            node: Some(node::Node::TypeCast(Box::new(
                                                TypeCast {
                                                    arg: Some(Box::new(Node {
                                                        node: Some(node::Node::AConst(
                                                            Box::new(AConst {
                                                                val: Some(Box::new(Node {
                                                                    node: Some(
                                                                        node::Node::String(
                                                                            String2 {
                                                                                str:
                                                                                    String::from(
                                                                                        "f",
                                                                                    ),
                                                                            },
                                                                        ),
                                                                    ),
                                                                })),
                                                                location: 415,
                                                            }),
                                                        )),
                                                    })),
                                                    type_name: Some(TypeName {
                                                        names: vec![
                                                            Node {
                                                                node: Some(node::Node::String(
                                                                    String2 {
                                                                        str: String::from(
                                                                            "pg_catalog",
                                                                        ),
                                                                    },
                                                                )),
                                                            },
                                                            Node {
                                                                node: Some(node::Node::String(
                                                                    String2 {
                                                                        str: String::from(
                                                                            "bool",
                                                                        ),
                                                                    },
                                                                )),
                                                            },
                                                        ],
                                                        type_oid: 0,
                                                        setof: false,
                                                        pct_type: false,
                                                        typmods: vec![],
                                                        typemod: -1,
                                                        array_bounds: vec![],
                                                        location: -1,
                                                    }),
                                                    location: -1,
                                                },
                                            ))),
                                        })),
                                        location: 413,
                                    }))),
                                },
                            ],
                            location: 398,
                        }))),
                    })),
                    group_clause: vec![],
                    having_clause: None,
                    window_clause: vec![],
                    values_lists: vec![],
                    sort_clause: vec![],
                    limit_offset: None,
                    limit_count: None,
                    limit_option: LimitOption::Default as i32,
                    locking_clause: vec![],
                    with_clause: Some(WithClause {
                        ctes: inner_cte,
                        recursive: false,
                        location: 24,
                    }),
                    op: SetOperation::SetopNone as i32,
                    all: false,
                    larg: None,
                    rarg: None,
                }))),
            })),
            location: 5,
            cterecursive: false,
            cterefcount: 0,
            ctecolnames: vec![],
            ctecoltypes: vec![],
            ctecoltypmods: vec![],
            ctecolcollations: vec![],
        }))),
    }];
    let expected_result = ParseResult {
        version: 130003,
        stmts: vec![RawStmt {
            stmt: Some(Box::new(Node {
                node: Some(node::Node::SelectStmt(Box::new(SelectStmt {
                    distinct_clause: vec![],
                    into_clause: None,

                    target_list: vec![Node {
                        node: Some(node::Node::ResTarget(Box::new(ResTarget {
                            name: String::from(""),
                            indirection: vec![],
                            val: Some(Box::new(Node {
                                node: Some(node::Node::ColumnRef(ColumnRef {
                                    fields: vec![Node {
                                        node: Some(node::Node::String(String2 {
                                            str: String::from("column1"),
                                        })),
                                    }],
                                    location: 430,
                                })),
                            })),
                            location: 430,
                        }))),
                    }],
                    from_clause: vec![Node {
                        node: Some(node::Node::RangeVar(RangeVar {
                            catalogname: String::from(""),
                            schemaname: String::from(""),
                            relname: String::from("table_name"),
                            inh: true,
                            relpersistence: String::from("p"),
                            alias: None,
                            location: 443,
                        })),
                    }],
                    where_clause: Some(Box::new(Node {
                        node: Some(node::Node::AExpr(Box::new(AExpr {
                            kind: AExprKind::AexprOp as i32,
                            name: vec![Node {
                                node: Some(node::Node::String(String2 {
                                    str: String::from(">"),
                                })),
                            }],
                            lexpr: Some(Box::new(Node {
                                node: Some(node::Node::ColumnRef(ColumnRef {
                                    fields: vec![Node {
                                        node: Some(node::Node::String(String2 {
                                            str: String::from("column2"),
                                        })),
                                    }],
                                    location: 460,
                                })),
                            })),
                            rexpr: Some(Box::new(Node {
                                node: Some(node::Node::AConst(Box::new(AConst {
                                    val: Some(Box::new(Node {
                                        node: Some(node::Node::Integer(Integer {
                                            ival: 9000,
                                        })),
                                    })),
                                    location: 470,
                                }))),
                            })),
                            location: 468,
                        }))),
                    })),
                    group_clause: vec![],
                    having_clause: None,
                    window_clause: vec![],
                    values_lists: vec![],
                    sort_clause: vec![],
                    limit_offset: None,
                    limit_count: None,
                    limit_option: LimitOption::Default as i32,
                    locking_clause: vec![],
                    with_clause: Some(WithClause {
                        ctes: outer_cte,
                        recursive: false,
                        location: 0,
                    }),
                    op: SetOperation::SetopNone as i32,
                    all: false,
                    larg: None,
                    rarg: None,
                }))),
            })),
            stmt_location: 0,
            stmt_len: 0,
        }],
    };

}
#[derive(Clone, PartialEq)]
pub struct ParseResult {

    pub version: i32,

    pub stmts: Vec<RawStmt>,
}
#[derive(Clone, PartialEq)]
pub struct ScanResult {

    pub version: i32,

    pub tokens: Vec<ScanToken>,
}
#[derive(Clone, PartialEq)]
pub struct Node {
    pub node: ::core::option::Option<node::Node>,
}
/// Nested message and enum types in `Node`.
pub mod node {
    #[derive(Clone, PartialEq)]
    pub enum Node {

        Alias(super::Alias),

        RangeVar(super::RangeVar),

        TableFunc(Box<super::TableFunc>),

        Expr(super::Expr),

        Var(Box<super::Var>),

        Param(Box<super::Param>),

        Aggref(Box<super::Aggref>),

        GroupingFunc(Box<super::GroupingFunc>),

        WindowFunc(Box<super::WindowFunc>),

        SubscriptingRef(Box<super::SubscriptingRef>),

        FuncExpr(Box<super::FuncExpr>),

        NamedArgExpr(Box<super::NamedArgExpr>),

        OpExpr(Box<super::OpExpr>),

        DistinctExpr(Box<super::DistinctExpr>),

        NullIfExpr(Box<super::NullIfExpr>),

        ScalarArrayOpExpr(Box<super::ScalarArrayOpExpr>),

        BoolExpr(Box<super::BoolExpr>),

        SubLink(Box<super::SubLink>),

        SubPlan(Box<super::SubPlan>),

        AlternativeSubPlan(Box<super::AlternativeSubPlan>),

        FieldSelect(Box<super::FieldSelect>),

        FieldStore(Box<super::FieldStore>),

        RelabelType(Box<super::RelabelType>),

        CoerceViaIo(Box<super::CoerceViaIo>),

        ArrayCoerceExpr(Box<super::ArrayCoerceExpr>),

        ConvertRowtypeExpr(Box<super::ConvertRowtypeExpr>),

        CollateExpr(Box<super::CollateExpr>),

        CaseExpr(Box<super::CaseExpr>),

        CaseWhen(Box<super::CaseWhen>),

        CaseTestExpr(Box<super::CaseTestExpr>),

        ArrayExpr(Box<super::ArrayExpr>),

        RowExpr(Box<super::RowExpr>),

        RowCompareExpr(Box<super::RowCompareExpr>),

        CoalesceExpr(Box<super::CoalesceExpr>),

        MinMaxExpr(Box<super::MinMaxExpr>),

        SqlvalueFunction(Box<super::SqlValueFunction>),

        XmlExpr(Box<super::XmlExpr>),

        NullTest(Box<super::NullTest>),

        BooleanTest(Box<super::BooleanTest>),

        CoerceToDomain(Box<super::CoerceToDomain>),

        CoerceToDomainValue(Box<super::CoerceToDomainValue>),

        SetToDefault(Box<super::SetToDefault>),

        CurrentOfExpr(Box<super::CurrentOfExpr>),

        NextValueExpr(Box<super::NextValueExpr>),

        InferenceElem(Box<super::InferenceElem>),

        TargetEntry(Box<super::TargetEntry>),

        RangeTblRef(super::RangeTblRef),

        JoinExpr(Box<super::JoinExpr>),

        FromExpr(Box<super::FromExpr>),

        OnConflictExpr(Box<super::OnConflictExpr>),

        IntoClause(Box<super::IntoClause>),

        RawStmt(Box<super::RawStmt>),

        Query(Box<super::Query>),

        InsertStmt(Box<super::InsertStmt>),

        DeleteStmt(Box<super::DeleteStmt>),

        UpdateStmt(Box<super::UpdateStmt>),

        SelectStmt(Box<super::SelectStmt>),

        AlterTableStmt(super::AlterTableStmt),

        AlterTableCmd(Box<super::AlterTableCmd>),

        AlterDomainStmt(Box<super::AlterDomainStmt>),

        SetOperationStmt(Box<super::SetOperationStmt>),

        GrantStmt(super::GrantStmt),

        GrantRoleStmt(super::GrantRoleStmt),

        AlterDefaultPrivilegesStmt(super::AlterDefaultPrivilegesStmt),

        ClosePortalStmt(super::ClosePortalStmt),

        ClusterStmt(super::ClusterStmt),

        CopyStmt(Box<super::CopyStmt>),

        CreateStmt(super::CreateStmt),

        DefineStmt(super::DefineStmt),

        DropStmt(super::DropStmt),

        TruncateStmt(super::TruncateStmt),

        CommentStmt(Box<super::CommentStmt>),

        FetchStmt(super::FetchStmt),

        IndexStmt(Box<super::IndexStmt>),

        CreateFunctionStmt(super::CreateFunctionStmt),

        AlterFunctionStmt(super::AlterFunctionStmt),

        DoStmt(super::DoStmt),

        RenameStmt(Box<super::RenameStmt>),

        RuleStmt(Box<super::RuleStmt>),

        NotifyStmt(super::NotifyStmt),

        ListenStmt(super::ListenStmt),

        UnlistenStmt(super::UnlistenStmt),

        TransactionStmt(super::TransactionStmt),

        ViewStmt(Box<super::ViewStmt>),

        LoadStmt(super::LoadStmt),

        CreateDomainStmt(Box<super::CreateDomainStmt>),

        CreatedbStmt(super::CreatedbStmt),

        DropdbStmt(super::DropdbStmt),

        VacuumStmt(super::VacuumStmt),

        ExplainStmt(Box<super::ExplainStmt>),

        CreateTableAsStmt(Box<super::CreateTableAsStmt>),

        CreateSeqStmt(super::CreateSeqStmt),

        AlterSeqStmt(super::AlterSeqStmt),

        VariableSetStmt(super::VariableSetStmt),

        VariableShowStmt(super::VariableShowStmt),

        DiscardStmt(super::DiscardStmt),

        CreateTrigStmt(Box<super::CreateTrigStmt>),

        CreatePlangStmt(super::CreatePLangStmt),

        CreateRoleStmt(super::CreateRoleStmt),

        AlterRoleStmt(super::AlterRoleStmt),

        DropRoleStmt(super::DropRoleStmt),

        LockStmt(super::LockStmt),

        ConstraintsSetStmt(super::ConstraintsSetStmt),

        ReindexStmt(super::ReindexStmt),

        CheckPointStmt(super::CheckPointStmt),

        CreateSchemaStmt(super::CreateSchemaStmt),

        AlterDatabaseStmt(super::AlterDatabaseStmt),

        AlterDatabaseSetStmt(super::AlterDatabaseSetStmt),

        AlterRoleSetStmt(super::AlterRoleSetStmt),

        CreateConversionStmt(super::CreateConversionStmt),

        CreateCastStmt(super::CreateCastStmt),

        CreateOpClassStmt(super::CreateOpClassStmt),

        CreateOpFamilyStmt(super::CreateOpFamilyStmt),

        AlterOpFamilyStmt(super::AlterOpFamilyStmt),

        PrepareStmt(Box<super::PrepareStmt>),

        ExecuteStmt(super::ExecuteStmt),

        DeallocateStmt(super::DeallocateStmt),

        DeclareCursorStmt(Box<super::DeclareCursorStmt>),

        CreateTableSpaceStmt(super::CreateTableSpaceStmt),

        DropTableSpaceStmt(super::DropTableSpaceStmt),

        AlterObjectDependsStmt(Box<super::AlterObjectDependsStmt>),

        AlterObjectSchemaStmt(Box<super::AlterObjectSchemaStmt>),

        AlterOwnerStmt(Box<super::AlterOwnerStmt>),

        AlterOperatorStmt(super::AlterOperatorStmt),

        AlterTypeStmt(super::AlterTypeStmt),

        DropOwnedStmt(super::DropOwnedStmt),

        ReassignOwnedStmt(super::ReassignOwnedStmt),

        CompositeTypeStmt(super::CompositeTypeStmt),

        CreateEnumStmt(super::CreateEnumStmt),

        CreateRangeStmt(super::CreateRangeStmt),

        AlterEnumStmt(super::AlterEnumStmt),

        AlterTsdictionaryStmt(super::AlterTsDictionaryStmt),

        AlterTsconfigurationStmt(super::AlterTsConfigurationStmt),

        CreateFdwStmt(super::CreateFdwStmt),

        AlterFdwStmt(super::AlterFdwStmt),

        CreateForeignServerStmt(super::CreateForeignServerStmt),

        AlterForeignServerStmt(super::AlterForeignServerStmt),

        CreateUserMappingStmt(super::CreateUserMappingStmt),

        AlterUserMappingStmt(super::AlterUserMappingStmt),

        DropUserMappingStmt(super::DropUserMappingStmt),

        AlterTableSpaceOptionsStmt(super::AlterTableSpaceOptionsStmt),

        AlterTableMoveAllStmt(super::AlterTableMoveAllStmt),

        SecLabelStmt(Box<super::SecLabelStmt>),

        CreateForeignTableStmt(super::CreateForeignTableStmt),

        ImportForeignSchemaStmt(super::ImportForeignSchemaStmt),

        CreateExtensionStmt(super::CreateExtensionStmt),

        AlterExtensionStmt(super::AlterExtensionStmt),

        AlterExtensionContentsStmt(Box<super::AlterExtensionContentsStmt>),

        CreateEventTrigStmt(super::CreateEventTrigStmt),

        AlterEventTrigStmt(super::AlterEventTrigStmt),

        RefreshMatViewStmt(super::RefreshMatViewStmt),

        ReplicaIdentityStmt(super::ReplicaIdentityStmt),

        AlterSystemStmt(super::AlterSystemStmt),

        CreatePolicyStmt(Box<super::CreatePolicyStmt>),

        AlterPolicyStmt(Box<super::AlterPolicyStmt>),

        CreateTransformStmt(super::CreateTransformStmt),

        CreateAmStmt(super::CreateAmStmt),

        CreatePublicationStmt(super::CreatePublicationStmt),

        AlterPublicationStmt(super::AlterPublicationStmt),

        CreateSubscriptionStmt(super::CreateSubscriptionStmt),

        AlterSubscriptionStmt(super::AlterSubscriptionStmt),

        DropSubscriptionStmt(super::DropSubscriptionStmt),

        CreateStatsStmt(super::CreateStatsStmt),

        AlterCollationStmt(super::AlterCollationStmt),

        CallStmt(Box<super::CallStmt>),

        AlterStatsStmt(super::AlterStatsStmt),

        AExpr(Box<super::AExpr>),

        ColumnRef(super::ColumnRef),

        ParamRef(super::ParamRef),

        AConst(Box<super::AConst>),

        FuncCall(Box<super::FuncCall>),

        AStar(super::AStar),

        AIndices(Box<super::AIndices>),

        AIndirection(Box<super::AIndirection>),

        AArrayExpr(super::AArrayExpr),

        ResTarget(Box<super::ResTarget>),

        MultiAssignRef(Box<super::MultiAssignRef>),

        TypeCast(Box<super::TypeCast>),

        CollateClause(Box<super::CollateClause>),

        SortBy(Box<super::SortBy>),

        WindowDef(Box<super::WindowDef>),

        RangeSubselect(Box<super::RangeSubselect>),

        RangeFunction(super::RangeFunction),

        RangeTableSample(Box<super::RangeTableSample>),

        RangeTableFunc(Box<super::RangeTableFunc>),

        RangeTableFuncCol(Box<super::RangeTableFuncCol>),

        TypeName(super::TypeName),

        ColumnDef(Box<super::ColumnDef>),

        IndexElem(Box<super::IndexElem>),

        Constraint(Box<super::Constraint>),

        DefElem(Box<super::DefElem>),

        RangeTblEntry(Box<super::RangeTblEntry>),

        RangeTblFunction(Box<super::RangeTblFunction>),

        TableSampleClause(Box<super::TableSampleClause>),

        WithCheckOption(Box<super::WithCheckOption>),

        SortGroupClause(super::SortGroupClause),

        GroupingSet(super::GroupingSet),

        WindowClause(Box<super::WindowClause>),

        ObjectWithArgs(super::ObjectWithArgs),

        AccessPriv(super::AccessPriv),

        CreateOpClassItem(super::CreateOpClassItem),

        TableLikeClause(super::TableLikeClause),

        FunctionParameter(Box<super::FunctionParameter>),

        LockingClause(super::LockingClause),

        RowMarkClause(super::RowMarkClause),

        XmlSerialize(Box<super::XmlSerialize>),

        WithClause(super::WithClause),

        InferClause(Box<super::InferClause>),

        OnConflictClause(Box<super::OnConflictClause>),

        CommonTableExpr(Box<super::CommonTableExpr>),

        RoleSpec(super::RoleSpec),

        TriggerTransition(super::TriggerTransition),

        PartitionElem(Box<super::PartitionElem>),

        PartitionSpec(super::PartitionSpec),

        PartitionBoundSpec(super::PartitionBoundSpec),

        PartitionRangeDatum(Box<super::PartitionRangeDatum>),

        PartitionCmd(super::PartitionCmd),

        VacuumRelation(super::VacuumRelation),

        InlineCodeBlock(super::InlineCodeBlock),

        CallContext(super::CallContext),

        Integer(super::Integer),

        Float(super::Float),

        String(super::String2),

        BitString(super::BitString),

        Null(super::Null),

        List(super::List),

        IntList(super::IntList),

        OidList(super::OidList),
    }
}
#[derive(Clone, PartialEq)]
pub struct Integer {
    /// machine integer

    pub ival: i32,
}
#[derive(Clone, PartialEq)]
pub struct Float {
    /// string

    pub str: String,
}
#[derive(Clone, PartialEq)]
pub struct String2 {
    /// string

    pub str: String,
}
#[derive(Clone, PartialEq)]
pub struct BitString {
    /// string

    pub str: String,
}
/// intentionally empty
#[derive(Clone, PartialEq)]
pub struct Null {}
#[derive(Clone, PartialEq)]
pub struct List {

    pub items: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct OidList {

    pub items: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct IntList {

    pub items: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct Alias {

    pub aliasname: String,

    pub colnames: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct RangeVar {

    pub catalogname: String,

    pub schemaname: String,

    pub relname: String,

    pub inh: bool,

    pub relpersistence: String,

    pub alias: ::core::option::Option<Alias>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct TableFunc {

    pub ns_uris: Vec<Node>,

    pub ns_names: Vec<Node>,

    pub docexpr: ::core::option::Option<Box<Node>>,

    pub rowexpr: ::core::option::Option<Box<Node>>,

    pub colnames: Vec<Node>,

    pub coltypes: Vec<Node>,

    pub coltypmods: Vec<Node>,

    pub colcollations: Vec<Node>,

    pub colexprs: Vec<Node>,

    pub coldefexprs: Vec<Node>,

    pub notnulls: Vec<u64>,

    pub ordinalitycol: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct Expr {}
#[derive(Clone, PartialEq)]
pub struct Var {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub varno: u32,

    pub varattno: i32,

    pub vartype: u32,

    pub vartypmod: i32,

    pub varcollid: u32,

    pub varlevelsup: u32,

    pub varnosyn: u32,

    pub varattnosyn: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct Param {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub paramkind: i32,

    pub paramid: i32,

    pub paramtype: u32,

    pub paramtypmod: i32,

    pub paramcollid: u32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct Aggref {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub aggfnoid: u32,

    pub aggtype: u32,

    pub aggcollid: u32,

    pub inputcollid: u32,

    pub aggtranstype: u32,

    pub aggargtypes: Vec<Node>,

    pub aggdirectargs: Vec<Node>,

    pub args: Vec<Node>,

    pub aggorder: Vec<Node>,

    pub aggdistinct: Vec<Node>,

    pub aggfilter: ::core::option::Option<Box<Node>>,

    pub aggstar: bool,

    pub aggvariadic: bool,

    pub aggkind: String,

    pub agglevelsup: u32,

    pub aggsplit: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct GroupingFunc {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub args: Vec<Node>,

    pub refs: Vec<Node>,

    pub cols: Vec<Node>,

    pub agglevelsup: u32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct WindowFunc {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub winfnoid: u32,

    pub wintype: u32,

    pub wincollid: u32,

    pub inputcollid: u32,

    pub args: Vec<Node>,

    pub aggfilter: ::core::option::Option<Box<Node>>,

    pub winref: u32,

    pub winstar: bool,

    pub winagg: bool,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct SubscriptingRef {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub refcontainertype: u32,

    pub refelemtype: u32,

    pub reftypmod: i32,

    pub refcollid: u32,

    pub refupperindexpr: Vec<Node>,

    pub reflowerindexpr: Vec<Node>,

    pub refexpr: ::core::option::Option<Box<Node>>,

    pub refassgnexpr: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct FuncExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub funcid: u32,

    pub funcresulttype: u32,

    pub funcretset: bool,

    pub funcvariadic: bool,

    pub funcformat: i32,

    pub funccollid: u32,

    pub inputcollid: u32,

    pub args: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct NamedArgExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub name: String,

    pub argnumber: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct OpExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub opno: u32,

    pub opfuncid: u32,

    pub opresulttype: u32,

    pub opretset: bool,

    pub opcollid: u32,

    pub inputcollid: u32,

    pub args: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct DistinctExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub opno: u32,

    pub opfuncid: u32,

    pub opresulttype: u32,

    pub opretset: bool,

    pub opcollid: u32,

    pub inputcollid: u32,

    pub args: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct NullIfExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub opno: u32,

    pub opfuncid: u32,

    pub opresulttype: u32,

    pub opretset: bool,

    pub opcollid: u32,

    pub inputcollid: u32,

    pub args: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct ScalarArrayOpExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub opno: u32,

    pub opfuncid: u32,

    pub use_or: bool,

    pub inputcollid: u32,

    pub args: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct BoolExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub boolop: i32,

    pub args: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct SubLink {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub sub_link_type: i32,

    pub sub_link_id: i32,

    pub testexpr: ::core::option::Option<Box<Node>>,

    pub oper_name: Vec<Node>,

    pub subselect: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct SubPlan {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub sub_link_type: i32,

    pub testexpr: ::core::option::Option<Box<Node>>,

    pub param_ids: Vec<Node>,

    pub plan_id: i32,

    pub plan_name: String,

    pub first_col_type: u32,

    pub first_col_typmod: i32,

    pub first_col_collation: u32,

    pub use_hash_table: bool,

    pub unknown_eq_false: bool,

    pub parallel_safe: bool,

    pub set_param: Vec<Node>,

    pub par_param: Vec<Node>,

    pub args: Vec<Node>,

    pub startup_cost: f64,

    pub per_call_cost: f64,
}
#[derive(Clone, PartialEq)]
pub struct AlternativeSubPlan {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub subplans: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct FieldSelect {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub fieldnum: i32,

    pub resulttype: u32,

    pub resulttypmod: i32,

    pub resultcollid: u32,
}
#[derive(Clone, PartialEq)]
pub struct FieldStore {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub newvals: Vec<Node>,

    pub fieldnums: Vec<Node>,

    pub resulttype: u32,
}
#[derive(Clone, PartialEq)]
pub struct RelabelType {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub resulttype: u32,

    pub resulttypmod: i32,

    pub resultcollid: u32,

    pub relabelformat: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CoerceViaIo {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub resulttype: u32,

    pub resultcollid: u32,

    pub coerceformat: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct ArrayCoerceExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub elemexpr: ::core::option::Option<Box<Node>>,

    pub resulttype: u32,

    pub resulttypmod: i32,

    pub resultcollid: u32,

    pub coerceformat: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct ConvertRowtypeExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub resulttype: u32,

    pub convertformat: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CollateExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub coll_oid: u32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CaseExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub casetype: u32,

    pub casecollid: u32,

    pub arg: ::core::option::Option<Box<Node>>,

    pub args: Vec<Node>,

    pub defresult: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CaseWhen {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub expr: ::core::option::Option<Box<Node>>,

    pub result: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CaseTestExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub type_id: u32,

    pub type_mod: i32,

    pub collation: u32,
}
#[derive(Clone, PartialEq)]
pub struct ArrayExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub array_typeid: u32,

    pub array_collid: u32,

    pub element_typeid: u32,

    pub elements: Vec<Node>,

    pub multidims: bool,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct RowExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub args: Vec<Node>,

    pub row_typeid: u32,

    pub row_format: i32,

    pub colnames: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct RowCompareExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub rctype: i32,

    pub opnos: Vec<Node>,

    pub opfamilies: Vec<Node>,

    pub inputcollids: Vec<Node>,

    pub largs: Vec<Node>,

    pub rargs: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CoalesceExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub coalescetype: u32,

    pub coalescecollid: u32,

    pub args: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct MinMaxExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub minmaxtype: u32,

    pub minmaxcollid: u32,

    pub inputcollid: u32,

    pub op: i32,

    pub args: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct SqlValueFunction {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub op: i32,

    pub r#type: u32,

    pub typmod: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct XmlExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub op: i32,

    pub name: String,

    pub named_args: Vec<Node>,

    pub arg_names: Vec<Node>,

    pub args: Vec<Node>,

    pub xmloption: i32,

    pub r#type: u32,

    pub typmod: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct NullTest {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub nulltesttype: i32,

    pub argisrow: bool,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct BooleanTest {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub booltesttype: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CoerceToDomain {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub arg: ::core::option::Option<Box<Node>>,

    pub resulttype: u32,

    pub resulttypmod: i32,

    pub resultcollid: u32,

    pub coercionformat: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CoerceToDomainValue {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub type_id: u32,

    pub type_mod: i32,

    pub collation: u32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct SetToDefault {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub type_id: u32,

    pub type_mod: i32,

    pub collation: u32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CurrentOfExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub cvarno: u32,

    pub cursor_name: String,

    pub cursor_param: i32,
}
#[derive(Clone, PartialEq)]
pub struct NextValueExpr {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub seqid: u32,

    pub type_id: u32,
}
#[derive(Clone, PartialEq)]
pub struct InferenceElem {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub expr: ::core::option::Option<Box<Node>>,

    pub infercollid: u32,

    pub inferopclass: u32,
}
#[derive(Clone, PartialEq)]
pub struct TargetEntry {

    pub xpr: ::core::option::Option<Box<Node>>,

    pub expr: ::core::option::Option<Box<Node>>,

    pub resno: i32,

    pub resname: String,

    pub ressortgroupref: u32,

    pub resorigtbl: u32,

    pub resorigcol: i32,

    pub resjunk: bool,
}
#[derive(Clone, PartialEq)]
pub struct RangeTblRef {

    pub rtindex: i32,
}
#[derive(Clone, PartialEq)]
pub struct JoinExpr {

    pub jointype: i32,

    pub is_natural: bool,

    pub larg: ::core::option::Option<Box<Node>>,

    pub rarg: ::core::option::Option<Box<Node>>,

    pub using_clause: Vec<Node>,

    pub quals: ::core::option::Option<Box<Node>>,

    pub alias: ::core::option::Option<Alias>,

    pub rtindex: i32,
}
#[derive(Clone, PartialEq)]
pub struct FromExpr {

    pub fromlist: Vec<Node>,

    pub quals: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct OnConflictExpr {

    pub action: i32,

    pub arbiter_elems: Vec<Node>,

    pub arbiter_where: ::core::option::Option<Box<Node>>,

    pub constraint: u32,

    pub on_conflict_set: Vec<Node>,

    pub on_conflict_where: ::core::option::Option<Box<Node>>,

    pub excl_rel_index: i32,

    pub excl_rel_tlist: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct IntoClause {

    pub rel: ::core::option::Option<RangeVar>,

    pub col_names: Vec<Node>,

    pub access_method: String,

    pub options: Vec<Node>,

    pub on_commit: i32,

    pub table_space_name: String,

    pub view_query: ::core::option::Option<Box<Node>>,

    pub skip_data: bool,
}
#[derive(Clone, PartialEq)]
pub struct RawStmt {

    pub stmt: ::core::option::Option<Box<Node>>,

    pub stmt_location: i32,

    pub stmt_len: i32,
}
#[derive(Clone, PartialEq)]
pub struct Query {

    pub command_type: i32,

    pub query_source: i32,

    pub can_set_tag: bool,

    pub utility_stmt: ::core::option::Option<Box<Node>>,

    pub result_relation: i32,

    pub has_aggs: bool,

    pub has_window_funcs: bool,

    pub has_target_srfs: bool,

    pub has_sub_links: bool,

    pub has_distinct_on: bool,

    pub has_recursive: bool,

    pub has_modifying_cte: bool,

    pub has_for_update: bool,

    pub has_row_security: bool,

    pub cte_list: Vec<Node>,

    pub rtable: Vec<Node>,

    pub jointree: ::core::option::Option<Box<FromExpr>>,

    pub target_list: Vec<Node>,

    pub r#override: i32,

    pub on_conflict: ::core::option::Option<Box<OnConflictExpr>>,

    pub returning_list: Vec<Node>,

    pub group_clause: Vec<Node>,

    pub grouping_sets: Vec<Node>,

    pub having_qual: ::core::option::Option<Box<Node>>,

    pub window_clause: Vec<Node>,

    pub distinct_clause: Vec<Node>,

    pub sort_clause: Vec<Node>,

    pub limit_offset: ::core::option::Option<Box<Node>>,

    pub limit_count: ::core::option::Option<Box<Node>>,

    pub limit_option: i32,

    pub row_marks: Vec<Node>,

    pub set_operations: ::core::option::Option<Box<Node>>,

    pub constraint_deps: Vec<Node>,

    pub with_check_options: Vec<Node>,

    pub stmt_location: i32,

    pub stmt_len: i32,
}
#[derive(Clone, PartialEq)]
pub struct InsertStmt {

    pub relation: ::core::option::Option<RangeVar>,

    pub cols: Vec<Node>,

    pub select_stmt: ::core::option::Option<Box<Node>>,

    pub on_conflict_clause: ::core::option::Option<Box<OnConflictClause>>,

    pub returning_list: Vec<Node>,

    pub with_clause: ::core::option::Option<WithClause>,

    pub r#override: i32,
}
#[derive(Clone, PartialEq)]
pub struct DeleteStmt {

    pub relation: ::core::option::Option<RangeVar>,

    pub using_clause: Vec<Node>,

    pub where_clause: ::core::option::Option<Box<Node>>,

    pub returning_list: Vec<Node>,

    pub with_clause: ::core::option::Option<WithClause>,
}
#[derive(Clone, PartialEq)]
pub struct UpdateStmt {

    pub relation: ::core::option::Option<RangeVar>,

    pub target_list: Vec<Node>,

    pub where_clause: ::core::option::Option<Box<Node>>,

    pub from_clause: Vec<Node>,

    pub returning_list: Vec<Node>,

    pub with_clause: ::core::option::Option<WithClause>,
}
#[derive(Clone, PartialEq)]
pub struct SelectStmt {

    pub distinct_clause: Vec<Node>,

    pub into_clause: ::core::option::Option<Box<IntoClause>>,

    pub target_list: Vec<Node>,

    pub from_clause: Vec<Node>,

    pub where_clause: ::core::option::Option<Box<Node>>,

    pub group_clause: Vec<Node>,

    pub having_clause: ::core::option::Option<Box<Node>>,

    pub window_clause: Vec<Node>,

    pub values_lists: Vec<Node>,

    pub sort_clause: Vec<Node>,

    pub limit_offset: ::core::option::Option<Box<Node>>,

    pub limit_count: ::core::option::Option<Box<Node>>,

    pub limit_option: i32,

    pub locking_clause: Vec<Node>,

    pub with_clause: ::core::option::Option<WithClause>,

    pub op: i32,

    pub all: bool,

    pub larg: ::core::option::Option<Box<SelectStmt>>,

    pub rarg: ::core::option::Option<Box<SelectStmt>>,
}
#[derive(Clone, PartialEq)]
pub struct AlterTableStmt {

    pub relation: ::core::option::Option<RangeVar>,

    pub cmds: Vec<Node>,

    pub relkind: i32,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterTableCmd {

    pub subtype: i32,

    pub name: String,

    pub num: i32,

    pub newowner: ::core::option::Option<RoleSpec>,

    pub def: ::core::option::Option<Box<Node>>,

    pub behavior: i32,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterDomainStmt {

    pub subtype: String,

    pub type_name: Vec<Node>,

    pub name: String,

    pub def: ::core::option::Option<Box<Node>>,

    pub behavior: i32,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct SetOperationStmt {

    pub op: i32,

    pub all: bool,

    pub larg: ::core::option::Option<Box<Node>>,

    pub rarg: ::core::option::Option<Box<Node>>,

    pub col_types: Vec<Node>,

    pub col_typmods: Vec<Node>,

    pub col_collations: Vec<Node>,

    pub group_clauses: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct GrantStmt {

    pub is_grant: bool,

    pub targtype: i32,

    pub objtype: i32,

    pub objects: Vec<Node>,

    pub privileges: Vec<Node>,

    pub grantees: Vec<Node>,

    pub grant_option: bool,

    pub behavior: i32,
}
#[derive(Clone, PartialEq)]
pub struct GrantRoleStmt {

    pub granted_roles: Vec<Node>,

    pub grantee_roles: Vec<Node>,

    pub is_grant: bool,

    pub admin_opt: bool,

    pub grantor: ::core::option::Option<RoleSpec>,

    pub behavior: i32,
}
#[derive(Clone, PartialEq)]
pub struct AlterDefaultPrivilegesStmt {

    pub options: Vec<Node>,

    pub action: ::core::option::Option<GrantStmt>,
}
#[derive(Clone, PartialEq)]
pub struct ClosePortalStmt {

    pub portalname: String,
}
#[derive(Clone, PartialEq)]
pub struct ClusterStmt {

    pub relation: ::core::option::Option<RangeVar>,

    pub indexname: String,

    pub options: i32,
}
#[derive(Clone, PartialEq)]
pub struct CopyStmt {

    pub relation: ::core::option::Option<RangeVar>,

    pub query: ::core::option::Option<Box<Node>>,

    pub attlist: Vec<Node>,

    pub is_from: bool,

    pub is_program: bool,

    pub filename: String,

    pub options: Vec<Node>,

    pub where_clause: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct CreateStmt {

    pub relation: ::core::option::Option<RangeVar>,

    pub table_elts: Vec<Node>,

    pub inh_relations: Vec<Node>,

    pub partbound: ::core::option::Option<PartitionBoundSpec>,

    pub partspec: ::core::option::Option<PartitionSpec>,

    pub of_typename: ::core::option::Option<TypeName>,

    pub constraints: Vec<Node>,

    pub options: Vec<Node>,

    pub oncommit: i32,

    pub tablespacename: String,

    pub access_method: String,

    pub if_not_exists: bool,
}
#[derive(Clone, PartialEq)]
pub struct DefineStmt {

    pub kind: i32,

    pub oldstyle: bool,

    pub defnames: Vec<Node>,

    pub args: Vec<Node>,

    pub definition: Vec<Node>,

    pub if_not_exists: bool,

    pub replace: bool,
}
#[derive(Clone, PartialEq)]
pub struct DropStmt {

    pub objects: Vec<Node>,

    pub remove_type: i32,

    pub behavior: i32,

    pub missing_ok: bool,

    pub concurrent: bool,
}
#[derive(Clone, PartialEq)]
pub struct TruncateStmt {

    pub relations: Vec<Node>,

    pub restart_seqs: bool,

    pub behavior: i32,
}
#[derive(Clone, PartialEq)]
pub struct CommentStmt {

    pub objtype: i32,

    pub object: ::core::option::Option<Box<Node>>,

    pub comment: String,
}
#[derive(Clone, PartialEq)]
pub struct FetchStmt {

    pub direction: i32,

    pub how_many: i64,

    pub portalname: String,

    pub ismove: bool,
}
#[derive(Clone, PartialEq)]
pub struct IndexStmt {

    pub idxname: String,

    pub relation: ::core::option::Option<RangeVar>,

    pub access_method: String,

    pub table_space: String,

    pub index_params: Vec<Node>,

    pub index_including_params: Vec<Node>,

    pub options: Vec<Node>,

    pub where_clause: ::core::option::Option<Box<Node>>,

    pub exclude_op_names: Vec<Node>,

    pub idxcomment: String,

    pub index_oid: u32,

    pub old_node: u32,

    pub old_create_subid: u32,

    pub old_first_relfilenode_subid: u32,

    pub unique: bool,

    pub primary: bool,

    pub isconstraint: bool,

    pub deferrable: bool,

    pub initdeferred: bool,

    pub transformed: bool,

    pub concurrent: bool,

    pub if_not_exists: bool,

    pub reset_default_tblspc: bool,
}
#[derive(Clone, PartialEq)]
pub struct CreateFunctionStmt {

    pub is_procedure: bool,

    pub replace: bool,

    pub funcname: Vec<Node>,

    pub parameters: Vec<Node>,

    pub return_type: ::core::option::Option<TypeName>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterFunctionStmt {

    pub objtype: i32,

    pub func: ::core::option::Option<ObjectWithArgs>,

    pub actions: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct DoStmt {

    pub args: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct RenameStmt {

    pub rename_type: i32,

    pub relation_type: i32,

    pub relation: ::core::option::Option<RangeVar>,

    pub object: ::core::option::Option<Box<Node>>,

    pub subname: String,

    pub newname: String,

    pub behavior: i32,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct RuleStmt {

    pub relation: ::core::option::Option<RangeVar>,

    pub rulename: String,

    pub where_clause: ::core::option::Option<Box<Node>>,

    pub event: i32,

    pub instead: bool,

    pub actions: Vec<Node>,

    pub replace: bool,
}
#[derive(Clone, PartialEq)]
pub struct NotifyStmt {

    pub conditionname: String,

    pub payload: String,
}
#[derive(Clone, PartialEq)]
pub struct ListenStmt {

    pub conditionname: String,
}
#[derive(Clone, PartialEq)]
pub struct UnlistenStmt {

    pub conditionname: String,
}
#[derive(Clone, PartialEq)]
pub struct TransactionStmt {

    pub kind: i32,

    pub options: Vec<Node>,

    pub savepoint_name: String,

    pub gid: String,

    pub chain: bool,
}
#[derive(Clone, PartialEq)]
pub struct ViewStmt {

    pub view: ::core::option::Option<RangeVar>,

    pub aliases: Vec<Node>,

    pub query: ::core::option::Option<Box<Node>>,

    pub replace: bool,

    pub options: Vec<Node>,

    pub with_check_option: i32,
}
#[derive(Clone, PartialEq)]
pub struct LoadStmt {

    pub filename: String,
}
#[derive(Clone, PartialEq)]
pub struct CreateDomainStmt {

    pub domainname: Vec<Node>,

    pub type_name: ::core::option::Option<TypeName>,

    pub coll_clause: ::core::option::Option<Box<CollateClause>>,

    pub constraints: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CreatedbStmt {

    pub dbname: String,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct DropdbStmt {

    pub dbname: String,

    pub missing_ok: bool,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct VacuumStmt {

    pub options: Vec<Node>,

    pub rels: Vec<Node>,

    pub is_vacuumcmd: bool,
}
#[derive(Clone, PartialEq)]
pub struct ExplainStmt {

    pub query: ::core::option::Option<Box<Node>>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CreateTableAsStmt {

    pub query: ::core::option::Option<Box<Node>>,

    pub into: ::core::option::Option<Box<IntoClause>>,

    pub relkind: i32,

    pub is_select_into: bool,

    pub if_not_exists: bool,
}
#[derive(Clone, PartialEq)]
pub struct CreateSeqStmt {

    pub sequence: ::core::option::Option<RangeVar>,

    pub options: Vec<Node>,

    pub owner_id: u32,

    pub for_identity: bool,

    pub if_not_exists: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterSeqStmt {

    pub sequence: ::core::option::Option<RangeVar>,

    pub options: Vec<Node>,

    pub for_identity: bool,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct VariableSetStmt {

    pub kind: i32,

    pub name: String,

    pub args: Vec<Node>,

    pub is_local: bool,
}
#[derive(Clone, PartialEq)]
pub struct VariableShowStmt {

    pub name: String,
}
#[derive(Clone, PartialEq)]
pub struct DiscardStmt {

    pub target: i32,
}
#[derive(Clone, PartialEq)]
pub struct CreateTrigStmt {

    pub trigname: String,

    pub relation: ::core::option::Option<RangeVar>,

    pub funcname: Vec<Node>,

    pub args: Vec<Node>,

    pub row: bool,

    pub timing: i32,

    pub events: i32,

    pub columns: Vec<Node>,

    pub when_clause: ::core::option::Option<Box<Node>>,

    pub isconstraint: bool,

    pub transition_rels: Vec<Node>,

    pub deferrable: bool,

    pub initdeferred: bool,

    pub constrrel: ::core::option::Option<RangeVar>,
}
#[derive(Clone, PartialEq)]
pub struct CreatePLangStmt {

    pub replace: bool,

    pub plname: String,

    pub plhandler: Vec<Node>,

    pub plinline: Vec<Node>,

    pub plvalidator: Vec<Node>,

    pub pltrusted: bool,
}
#[derive(Clone, PartialEq)]
pub struct CreateRoleStmt {

    pub stmt_type: i32,

    pub role: String,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterRoleStmt {

    pub role: ::core::option::Option<RoleSpec>,

    pub options: Vec<Node>,

    pub action: i32,
}
#[derive(Clone, PartialEq)]
pub struct DropRoleStmt {

    pub roles: Vec<Node>,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct LockStmt {

    pub relations: Vec<Node>,

    pub mode: i32,

    pub nowait: bool,
}
#[derive(Clone, PartialEq)]
pub struct ConstraintsSetStmt {

    pub constraints: Vec<Node>,

    pub deferred: bool,
}
#[derive(Clone, PartialEq)]
pub struct ReindexStmt {

    pub kind: i32,

    pub relation: ::core::option::Option<RangeVar>,

    pub name: String,

    pub options: i32,

    pub concurrent: bool,
}
#[derive(Clone, PartialEq)]
pub struct CheckPointStmt {}
#[derive(Clone, PartialEq)]
pub struct CreateSchemaStmt {

    pub schemaname: String,

    pub authrole: ::core::option::Option<RoleSpec>,

    pub schema_elts: Vec<Node>,

    pub if_not_exists: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterDatabaseStmt {

    pub dbname: String,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterDatabaseSetStmt {

    pub dbname: String,

    pub setstmt: ::core::option::Option<VariableSetStmt>,
}
#[derive(Clone, PartialEq)]
pub struct AlterRoleSetStmt {

    pub role: ::core::option::Option<RoleSpec>,

    pub database: String,

    pub setstmt: ::core::option::Option<VariableSetStmt>,
}
#[derive(Clone, PartialEq)]
pub struct CreateConversionStmt {

    pub conversion_name: Vec<Node>,

    pub for_encoding_name: String,

    pub to_encoding_name: String,

    pub func_name: Vec<Node>,

    pub def: bool,
}
#[derive(Clone, PartialEq)]
pub struct CreateCastStmt {

    pub sourcetype: ::core::option::Option<TypeName>,

    pub targettype: ::core::option::Option<TypeName>,

    pub func: ::core::option::Option<ObjectWithArgs>,

    pub context: i32,

    pub inout: bool,
}
#[derive(Clone, PartialEq)]
pub struct CreateOpClassStmt {

    pub opclassname: Vec<Node>,

    pub opfamilyname: Vec<Node>,

    pub amname: String,

    pub datatype: ::core::option::Option<TypeName>,

    pub items: Vec<Node>,

    pub is_default: bool,
}
#[derive(Clone, PartialEq)]
pub struct CreateOpFamilyStmt {

    pub opfamilyname: Vec<Node>,

    pub amname: String,
}
#[derive(Clone, PartialEq)]
pub struct AlterOpFamilyStmt {

    pub opfamilyname: Vec<Node>,

    pub amname: String,

    pub is_drop: bool,

    pub items: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct PrepareStmt {

    pub name: String,

    pub argtypes: Vec<Node>,

    pub query: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct ExecuteStmt {

    pub name: String,

    pub params: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct DeallocateStmt {

    pub name: String,
}
#[derive(Clone, PartialEq)]
pub struct DeclareCursorStmt {

    pub portalname: String,

    pub options: i32,

    pub query: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct CreateTableSpaceStmt {

    pub tablespacename: String,

    pub owner: ::core::option::Option<RoleSpec>,

    pub location: String,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct DropTableSpaceStmt {

    pub tablespacename: String,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterObjectDependsStmt {

    pub object_type: i32,

    pub relation: ::core::option::Option<RangeVar>,

    pub object: ::core::option::Option<Box<Node>>,

    pub extname: ::core::option::Option<Box<Node>>,

    pub remove: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterObjectSchemaStmt {

    pub object_type: i32,

    pub relation: ::core::option::Option<RangeVar>,

    pub object: ::core::option::Option<Box<Node>>,

    pub newschema: String,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterOwnerStmt {

    pub object_type: i32,

    pub relation: ::core::option::Option<RangeVar>,

    pub object: ::core::option::Option<Box<Node>>,

    pub newowner: ::core::option::Option<RoleSpec>,
}
#[derive(Clone, PartialEq)]
pub struct AlterOperatorStmt {

    pub opername: ::core::option::Option<ObjectWithArgs>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterTypeStmt {

    pub type_name: Vec<Node>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct DropOwnedStmt {

    pub roles: Vec<Node>,

    pub behavior: i32,
}
#[derive(Clone, PartialEq)]
pub struct ReassignOwnedStmt {

    pub roles: Vec<Node>,

    pub newrole: ::core::option::Option<RoleSpec>,
}
#[derive(Clone, PartialEq)]
pub struct CompositeTypeStmt {

    pub typevar: ::core::option::Option<RangeVar>,

    pub coldeflist: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CreateEnumStmt {

    pub type_name: Vec<Node>,

    pub vals: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CreateRangeStmt {

    pub type_name: Vec<Node>,

    pub params: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterEnumStmt {

    pub type_name: Vec<Node>,

    pub old_val: String,

    pub new_val: String,

    pub new_val_neighbor: String,

    pub new_val_is_after: bool,

    pub skip_if_new_val_exists: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterTsDictionaryStmt {

    pub dictname: Vec<Node>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterTsConfigurationStmt {

    pub kind: i32,

    pub cfgname: Vec<Node>,

    pub tokentype: Vec<Node>,

    pub dicts: Vec<Node>,

    pub r#override: bool,

    pub replace: bool,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct CreateFdwStmt {

    pub fdwname: String,

    pub func_options: Vec<Node>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterFdwStmt {

    pub fdwname: String,

    pub func_options: Vec<Node>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CreateForeignServerStmt {

    pub servername: String,

    pub servertype: String,

    pub version: String,

    pub fdwname: String,

    pub if_not_exists: bool,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterForeignServerStmt {

    pub servername: String,

    pub version: String,

    pub options: Vec<Node>,

    pub has_version: bool,
}
#[derive(Clone, PartialEq)]
pub struct CreateUserMappingStmt {

    pub user: ::core::option::Option<RoleSpec>,

    pub servername: String,

    pub if_not_exists: bool,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterUserMappingStmt {

    pub user: ::core::option::Option<RoleSpec>,

    pub servername: String,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct DropUserMappingStmt {

    pub user: ::core::option::Option<RoleSpec>,

    pub servername: String,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterTableSpaceOptionsStmt {

    pub tablespacename: String,

    pub options: Vec<Node>,

    pub is_reset: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterTableMoveAllStmt {

    pub orig_tablespacename: String,

    pub objtype: i32,

    pub roles: Vec<Node>,

    pub new_tablespacename: String,

    pub nowait: bool,
}
#[derive(Clone, PartialEq)]
pub struct SecLabelStmt {

    pub objtype: i32,

    pub object: ::core::option::Option<Box<Node>>,

    pub provider: String,

    pub label: String,
}
#[derive(Clone, PartialEq)]
pub struct CreateForeignTableStmt {

    pub base_stmt: ::core::option::Option<CreateStmt>,

    pub servername: String,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct ImportForeignSchemaStmt {

    pub server_name: String,

    pub remote_schema: String,

    pub local_schema: String,

    pub list_type: i32,

    pub table_list: Vec<Node>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CreateExtensionStmt {

    pub extname: String,

    pub if_not_exists: bool,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterExtensionStmt {

    pub extname: String,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterExtensionContentsStmt {

    pub extname: String,

    pub action: i32,

    pub objtype: i32,

    pub object: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct CreateEventTrigStmt {

    pub trigname: String,

    pub eventname: String,

    pub whenclause: Vec<Node>,

    pub funcname: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterEventTrigStmt {

    pub trigname: String,

    pub tgenabled: String,
}
#[derive(Clone, PartialEq)]
pub struct RefreshMatViewStmt {

    pub concurrent: bool,

    pub skip_data: bool,

    pub relation: ::core::option::Option<RangeVar>,
}
#[derive(Clone, PartialEq)]
pub struct ReplicaIdentityStmt {

    pub identity_type: String,

    pub name: String,
}
#[derive(Clone, PartialEq)]
pub struct AlterSystemStmt {

    pub setstmt: ::core::option::Option<VariableSetStmt>,
}
#[derive(Clone, PartialEq)]
pub struct CreatePolicyStmt {

    pub policy_name: String,

    pub table: ::core::option::Option<RangeVar>,

    pub cmd_name: String,

    pub permissive: bool,

    pub roles: Vec<Node>,

    pub qual: ::core::option::Option<Box<Node>>,

    pub with_check: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct AlterPolicyStmt {

    pub policy_name: String,

    pub table: ::core::option::Option<RangeVar>,

    pub roles: Vec<Node>,

    pub qual: ::core::option::Option<Box<Node>>,

    pub with_check: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct CreateTransformStmt {

    pub replace: bool,

    pub type_name: ::core::option::Option<TypeName>,

    pub lang: String,

    pub fromsql: ::core::option::Option<ObjectWithArgs>,

    pub tosql: ::core::option::Option<ObjectWithArgs>,
}
#[derive(Clone, PartialEq)]
pub struct CreateAmStmt {

    pub amname: String,

    pub handler_name: Vec<Node>,

    pub amtype: String,
}
#[derive(Clone, PartialEq)]
pub struct CreatePublicationStmt {

    pub pubname: String,

    pub options: Vec<Node>,

    pub tables: Vec<Node>,

    pub for_all_tables: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterPublicationStmt {

    pub pubname: String,

    pub options: Vec<Node>,

    pub tables: Vec<Node>,

    pub for_all_tables: bool,

    pub table_action: i32,
}
#[derive(Clone, PartialEq)]
pub struct CreateSubscriptionStmt {

    pub subname: String,

    pub conninfo: String,

    pub publication: Vec<Node>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AlterSubscriptionStmt {

    pub kind: i32,

    pub subname: String,

    pub conninfo: String,

    pub publication: Vec<Node>,

    pub options: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct DropSubscriptionStmt {

    pub subname: String,

    pub missing_ok: bool,

    pub behavior: i32,
}
#[derive(Clone, PartialEq)]
pub struct CreateStatsStmt {

    pub defnames: Vec<Node>,

    pub stat_types: Vec<Node>,

    pub exprs: Vec<Node>,

    pub relations: Vec<Node>,

    pub stxcomment: String,

    pub if_not_exists: bool,
}
#[derive(Clone, PartialEq)]
pub struct AlterCollationStmt {

    pub collname: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CallStmt {

    pub funccall: ::core::option::Option<Box<FuncCall>>,

    pub funcexpr: ::core::option::Option<Box<FuncExpr>>,
}
#[derive(Clone, PartialEq)]
pub struct AlterStatsStmt {

    pub defnames: Vec<Node>,

    pub stxstattarget: i32,

    pub missing_ok: bool,
}
#[derive(Clone, PartialEq)]
pub struct AExpr {

    pub kind: i32,

    pub name: Vec<Node>,

    pub lexpr: ::core::option::Option<Box<Node>>,

    pub rexpr: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct ColumnRef {

    pub fields: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct ParamRef {

    pub number: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct AConst {

    pub val: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct FuncCall {

    pub funcname: Vec<Node>,

    pub args: Vec<Node>,

    pub agg_order: Vec<Node>,

    pub agg_filter: ::core::option::Option<Box<Node>>,

    pub agg_within_group: bool,

    pub agg_star: bool,

    pub agg_distinct: bool,

    pub func_variadic: bool,

    pub over: ::core::option::Option<Box<WindowDef>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct AStar {}
#[derive(Clone, PartialEq)]
pub struct AIndices {

    pub is_slice: bool,

    pub lidx: ::core::option::Option<Box<Node>>,

    pub uidx: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct AIndirection {

    pub arg: ::core::option::Option<Box<Node>>,

    pub indirection: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct AArrayExpr {

    pub elements: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct ResTarget {

    pub name: String,

    pub indirection: Vec<Node>,

    pub val: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct MultiAssignRef {

    pub source: ::core::option::Option<Box<Node>>,

    pub colno: i32,

    pub ncolumns: i32,
}
#[derive(Clone, PartialEq)]
pub struct TypeCast {

    pub arg: ::core::option::Option<Box<Node>>,

    pub type_name: ::core::option::Option<TypeName>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CollateClause {

    pub arg: ::core::option::Option<Box<Node>>,

    pub collname: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct SortBy {

    pub node: ::core::option::Option<Box<Node>>,

    pub sortby_dir: i32,

    pub sortby_nulls: i32,

    pub use_op: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct WindowDef {

    pub name: String,

    pub refname: String,

    pub partition_clause: Vec<Node>,

    pub order_clause: Vec<Node>,

    pub frame_options: i32,

    pub start_offset: ::core::option::Option<Box<Node>>,

    pub end_offset: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct RangeSubselect {

    pub lateral: bool,

    pub subquery: ::core::option::Option<Box<Node>>,

    pub alias: ::core::option::Option<Alias>,
}
#[derive(Clone, PartialEq)]
pub struct RangeFunction {

    pub lateral: bool,

    pub ordinality: bool,

    pub is_rowsfrom: bool,

    pub functions: Vec<Node>,

    pub alias: ::core::option::Option<Alias>,

    pub coldeflist: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct RangeTableSample {

    pub relation: ::core::option::Option<Box<Node>>,

    pub method: Vec<Node>,

    pub args: Vec<Node>,

    pub repeatable: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct RangeTableFunc {

    pub lateral: bool,

    pub docexpr: ::core::option::Option<Box<Node>>,

    pub rowexpr: ::core::option::Option<Box<Node>>,

    pub namespaces: Vec<Node>,

    pub columns: Vec<Node>,

    pub alias: ::core::option::Option<Alias>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct RangeTableFuncCol {

    pub colname: String,

    pub type_name: ::core::option::Option<TypeName>,

    pub for_ordinality: bool,

    pub is_not_null: bool,

    pub colexpr: ::core::option::Option<Box<Node>>,

    pub coldefexpr: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct TypeName {

    pub names: Vec<Node>,

    pub type_oid: u32,

    pub setof: bool,

    pub pct_type: bool,

    pub typmods: Vec<Node>,

    pub typemod: i32,

    pub array_bounds: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct ColumnDef {

    pub colname: String,

    pub type_name: ::core::option::Option<TypeName>,

    pub inhcount: i32,

    pub is_local: bool,

    pub is_not_null: bool,

    pub is_from_type: bool,

    pub storage: String,

    pub raw_default: ::core::option::Option<Box<Node>>,

    pub cooked_default: ::core::option::Option<Box<Node>>,

    pub identity: String,

    pub identity_sequence: ::core::option::Option<RangeVar>,

    pub generated: String,

    pub coll_clause: ::core::option::Option<Box<CollateClause>>,

    pub coll_oid: u32,

    pub constraints: Vec<Node>,

    pub fdwoptions: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct IndexElem {

    pub name: String,

    pub expr: ::core::option::Option<Box<Node>>,

    pub indexcolname: String,

    pub collation: Vec<Node>,

    pub opclass: Vec<Node>,

    pub opclassopts: Vec<Node>,

    pub ordering: i32,

    pub nulls_ordering: i32,
}
#[derive(Clone, PartialEq)]
pub struct Constraint {

    pub contype: i32,

    pub conname: String,

    pub deferrable: bool,

    pub initdeferred: bool,

    pub location: i32,

    pub is_no_inherit: bool,

    pub raw_expr: ::core::option::Option<Box<Node>>,

    pub cooked_expr: String,

    pub generated_when: String,

    pub keys: Vec<Node>,

    pub including: Vec<Node>,

    pub exclusions: Vec<Node>,

    pub options: Vec<Node>,

    pub indexname: String,

    pub indexspace: String,

    pub reset_default_tblspc: bool,

    pub access_method: String,

    pub where_clause: ::core::option::Option<Box<Node>>,

    pub pktable: ::core::option::Option<RangeVar>,

    pub fk_attrs: Vec<Node>,

    pub pk_attrs: Vec<Node>,

    pub fk_matchtype: String,

    pub fk_upd_action: String,

    pub fk_del_action: String,

    pub old_conpfeqop: Vec<Node>,

    pub old_pktable_oid: u32,

    pub skip_validation: bool,

    pub initially_valid: bool,
}
#[derive(Clone, PartialEq)]
pub struct DefElem {

    pub defnamespace: String,

    pub defname: String,

    pub arg: ::core::option::Option<Box<Node>>,

    pub defaction: i32,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct RangeTblEntry {

    pub rtekind: i32,

    pub relid: u32,

    pub relkind: String,

    pub rellockmode: i32,

    pub tablesample: ::core::option::Option<Box<TableSampleClause>>,

    pub subquery: ::core::option::Option<Box<Query>>,

    pub security_barrier: bool,

    pub jointype: i32,

    pub joinmergedcols: i32,

    pub joinaliasvars: Vec<Node>,

    pub joinleftcols: Vec<Node>,

    pub joinrightcols: Vec<Node>,

    pub functions: Vec<Node>,

    pub funcordinality: bool,

    pub tablefunc: ::core::option::Option<Box<TableFunc>>,

    pub values_lists: Vec<Node>,

    pub ctename: String,

    pub ctelevelsup: u32,

    pub self_reference: bool,

    pub coltypes: Vec<Node>,

    pub coltypmods: Vec<Node>,

    pub colcollations: Vec<Node>,

    pub enrname: String,

    pub enrtuples: f64,

    pub alias: ::core::option::Option<Alias>,

    pub eref: ::core::option::Option<Alias>,

    pub lateral: bool,

    pub inh: bool,

    pub in_from_cl: bool,

    pub required_perms: u32,

    pub check_as_user: u32,

    pub selected_cols: Vec<u64>,

    pub inserted_cols: Vec<u64>,

    pub updated_cols: Vec<u64>,

    pub extra_updated_cols: Vec<u64>,

    pub security_quals: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct RangeTblFunction {

    pub funcexpr: ::core::option::Option<Box<Node>>,

    pub funccolcount: i32,

    pub funccolnames: Vec<Node>,

    pub funccoltypes: Vec<Node>,

    pub funccoltypmods: Vec<Node>,

    pub funccolcollations: Vec<Node>,

    pub funcparams: Vec<u64>,
}
#[derive(Clone, PartialEq)]
pub struct TableSampleClause {

    pub tsmhandler: u32,

    pub args: Vec<Node>,

    pub repeatable: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct WithCheckOption {

    pub kind: i32,

    pub relname: String,

    pub polname: String,

    pub qual: ::core::option::Option<Box<Node>>,

    pub cascaded: bool,
}
#[derive(Clone, PartialEq)]
pub struct SortGroupClause {

    pub tle_sort_group_ref: u32,

    pub eqop: u32,

    pub sortop: u32,

    pub nulls_first: bool,

    pub hashable: bool,
}
#[derive(Clone, PartialEq)]
pub struct GroupingSet {

    pub kind: i32,

    pub content: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct WindowClause {

    pub name: String,

    pub refname: String,

    pub partition_clause: Vec<Node>,

    pub order_clause: Vec<Node>,

    pub frame_options: i32,

    pub start_offset: ::core::option::Option<Box<Node>>,

    pub end_offset: ::core::option::Option<Box<Node>>,

    pub start_in_range_func: u32,

    pub end_in_range_func: u32,

    pub in_range_coll: u32,

    pub in_range_asc: bool,

    pub in_range_nulls_first: bool,

    pub winref: u32,

    pub copied_order: bool,
}
#[derive(Clone, PartialEq)]
pub struct ObjectWithArgs {

    pub objname: Vec<Node>,

    pub objargs: Vec<Node>,

    pub args_unspecified: bool,
}
#[derive(Clone, PartialEq)]
pub struct AccessPriv {

    pub priv_name: String,

    pub cols: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct CreateOpClassItem {

    pub itemtype: i32,

    pub name: ::core::option::Option<ObjectWithArgs>,

    pub number: i32,

    pub order_family: Vec<Node>,

    pub class_args: Vec<Node>,

    pub storedtype: ::core::option::Option<TypeName>,
}
#[derive(Clone, PartialEq)]
pub struct TableLikeClause {

    pub relation: ::core::option::Option<RangeVar>,

    pub options: u32,

    pub relation_oid: u32,
}
#[derive(Clone, PartialEq)]
pub struct FunctionParameter {

    pub name: String,

    pub arg_type: ::core::option::Option<TypeName>,

    pub mode: i32,

    pub defexpr: ::core::option::Option<Box<Node>>,
}
#[derive(Clone, PartialEq)]
pub struct LockingClause {

    pub locked_rels: Vec<Node>,

    pub strength: i32,

    pub wait_policy: i32,
}
#[derive(Clone, PartialEq)]
pub struct RowMarkClause {

    pub rti: u32,

    pub strength: i32,

    pub wait_policy: i32,

    pub pushed_down: bool,
}
#[derive(Clone, PartialEq)]
pub struct XmlSerialize {

    pub xmloption: i32,

    pub expr: ::core::option::Option<Box<Node>>,

    pub type_name: ::core::option::Option<TypeName>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct WithClause {

    pub ctes: Vec<Node>,

    pub recursive: bool,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct InferClause {

    pub index_elems: Vec<Node>,

    pub where_clause: ::core::option::Option<Box<Node>>,

    pub conname: String,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct OnConflictClause {

    pub action: i32,

    pub infer: ::core::option::Option<Box<InferClause>>,

    pub target_list: Vec<Node>,

    pub where_clause: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct CommonTableExpr {

    pub ctename: String,

    pub aliascolnames: Vec<Node>,

    pub ctematerialized: i32,

    pub ctequery: ::core::option::Option<Box<Node>>,

    pub location: i32,

    pub cterecursive: bool,

    pub cterefcount: i32,

    pub ctecolnames: Vec<Node>,

    pub ctecoltypes: Vec<Node>,

    pub ctecoltypmods: Vec<Node>,

    pub ctecolcollations: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct RoleSpec {

    pub roletype: i32,

    pub rolename: String,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct TriggerTransition {

    pub name: String,

    pub is_new: bool,

    pub is_table: bool,
}
#[derive(Clone, PartialEq)]
pub struct PartitionElem {

    pub name: String,

    pub expr: ::core::option::Option<Box<Node>>,

    pub collation: Vec<Node>,

    pub opclass: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct PartitionSpec {

    pub strategy: String,

    pub part_params: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct PartitionBoundSpec {

    pub strategy: String,

    pub is_default: bool,

    pub modulus: i32,

    pub remainder: i32,

    pub listdatums: Vec<Node>,

    pub lowerdatums: Vec<Node>,

    pub upperdatums: Vec<Node>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct PartitionRangeDatum {

    pub kind: i32,

    pub value: ::core::option::Option<Box<Node>>,

    pub location: i32,
}
#[derive(Clone, PartialEq)]
pub struct PartitionCmd {

    pub name: ::core::option::Option<RangeVar>,

    pub bound: ::core::option::Option<PartitionBoundSpec>,
}
#[derive(Clone, PartialEq)]
pub struct VacuumRelation {

    pub relation: ::core::option::Option<RangeVar>,

    pub oid: u32,

    pub va_cols: Vec<Node>,
}
#[derive(Clone, PartialEq)]
pub struct InlineCodeBlock {

    pub source_text: String,

    pub lang_oid: u32,

    pub lang_is_trusted: bool,

    pub atomic: bool,
}
#[derive(Clone, PartialEq)]
pub struct CallContext {

    pub atomic: bool,
}
#[derive(Clone, PartialEq)]
pub struct ScanToken {

    pub start: i32,

    pub end: i32,

    pub token: i32,

    pub keyword_kind: i32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum OverridingKind {
    Undefined = 0,
    OverridingNotSet = 1,
    OverridingUserValue = 2,
    OverridingSystemValue = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum QuerySource {
    Undefined = 0,
    QsrcOriginal = 1,
    QsrcParser = 2,
    QsrcInsteadRule = 3,
    QsrcQualInsteadRule = 4,
    QsrcNonInsteadRule = 5,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum SortByDir {
    Undefined = 0,
    SortbyDefault = 1,
    SortbyAsc = 2,
    SortbyDesc = 3,
    SortbyUsing = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum SortByNulls {
    Undefined = 0,
    SortbyNullsDefault = 1,
    SortbyNullsFirst = 2,
    SortbyNullsLast = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum AExprKind {
    Undefined = 0,
    AexprOp = 1,
    AexprOpAny = 2,
    AexprOpAll = 3,
    AexprDistinct = 4,
    AexprNotDistinct = 5,
    AexprNullif = 6,
    AexprOf = 7,
    AexprIn = 8,
    AexprLike = 9,
    AexprIlike = 10,
    AexprSimilar = 11,
    AexprBetween = 12,
    AexprNotBetween = 13,
    AexprBetweenSym = 14,
    AexprNotBetweenSym = 15,
    AexprParen = 16,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum RoleSpecType {
    Undefined = 0,
    RolespecCstring = 1,
    RolespecCurrentUser = 2,
    RolespecSessionUser = 3,
    RolespecPublic = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum TableLikeOption {
    Undefined = 0,
    CreateTableLikeComments = 1,
    CreateTableLikeConstraints = 2,
    CreateTableLikeDefaults = 3,
    CreateTableLikeGenerated = 4,
    CreateTableLikeIdentity = 5,
    CreateTableLikeIndexes = 6,
    CreateTableLikeStatistics = 7,
    CreateTableLikeStorage = 8,
    CreateTableLikeAll = 9,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum DefElemAction {
    Undefined = 0,
    DefelemUnspec = 1,
    DefelemSet = 2,
    DefelemAdd = 3,
    DefelemDrop = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum PartitionRangeDatumKind {
    Undefined = 0,
    PartitionRangeDatumMinvalue = 1,
    PartitionRangeDatumValue = 2,
    PartitionRangeDatumMaxvalue = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum RteKind {
    RtekindUndefined = 0,
    RteRelation = 1,
    RteSubquery = 2,
    RteJoin = 3,
    RteFunction = 4,
    RteTablefunc = 5,
    RteValues = 6,
    RteCte = 7,
    RteNamedtuplestore = 8,
    RteResult = 9,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum WcoKind {
    WcokindUndefined = 0,
    WcoViewCheck = 1,
    WcoRlsInsertCheck = 2,
    WcoRlsUpdateCheck = 3,
    WcoRlsConflictCheck = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum GroupingSetKind {
    Undefined = 0,
    GroupingSetEmpty = 1,
    GroupingSetSimple = 2,
    GroupingSetRollup = 3,
    GroupingSetCube = 4,
    GroupingSetSets = 5,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum CteMaterialize {
    CtematerializeUndefined = 0,
    Default = 1,
    Always = 2,
    Never = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum SetOperation {
    Undefined = 0,
    SetopNone = 1,
    SetopUnion = 2,
    SetopIntersect = 3,
    SetopExcept = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum ObjectType {
    Undefined = 0,
    ObjectAccessMethod = 1,
    ObjectAggregate = 2,
    ObjectAmop = 3,
    ObjectAmproc = 4,
    ObjectAttribute = 5,
    ObjectCast = 6,
    ObjectColumn = 7,
    ObjectCollation = 8,
    ObjectConversion = 9,
    ObjectDatabase = 10,
    ObjectDefault = 11,
    ObjectDefacl = 12,
    ObjectDomain = 13,
    ObjectDomconstraint = 14,
    ObjectEventTrigger = 15,
    ObjectExtension = 16,
    ObjectFdw = 17,
    ObjectForeignServer = 18,
    ObjectForeignTable = 19,
    ObjectFunction = 20,
    ObjectIndex = 21,
    ObjectLanguage = 22,
    ObjectLargeobject = 23,
    ObjectMatview = 24,
    ObjectOpclass = 25,
    ObjectOperator = 26,
    ObjectOpfamily = 27,
    ObjectPolicy = 28,
    ObjectProcedure = 29,
    ObjectPublication = 30,
    ObjectPublicationRel = 31,
    ObjectRole = 32,
    ObjectRoutine = 33,
    ObjectRule = 34,
    ObjectSchema = 35,
    ObjectSequence = 36,
    ObjectSubscription = 37,
    ObjectStatisticExt = 38,
    ObjectTabconstraint = 39,
    ObjectTable = 40,
    ObjectTablespace = 41,
    ObjectTransform = 42,
    ObjectTrigger = 43,
    ObjectTsconfiguration = 44,
    ObjectTsdictionary = 45,
    ObjectTsparser = 46,
    ObjectTstemplate = 47,
    ObjectType = 48,
    ObjectUserMapping = 49,
    ObjectView = 50,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum DropBehavior {
    Undefined = 0,
    DropRestrict = 1,
    DropCascade = 2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum AlterTableType {
    Undefined = 0,
    AtAddColumn = 1,
    AtAddColumnRecurse = 2,
    AtAddColumnToView = 3,
    AtColumnDefault = 4,
    AtCookedColumnDefault = 5,
    AtDropNotNull = 6,
    AtSetNotNull = 7,
    AtDropExpression = 8,
    AtCheckNotNull = 9,
    AtSetStatistics = 10,
    AtSetOptions = 11,
    AtResetOptions = 12,
    AtSetStorage = 13,
    AtDropColumn = 14,
    AtDropColumnRecurse = 15,
    AtAddIndex = 16,
    AtReAddIndex = 17,
    AtAddConstraint = 18,
    AtAddConstraintRecurse = 19,
    AtReAddConstraint = 20,
    AtReAddDomainConstraint = 21,
    AtAlterConstraint = 22,
    AtValidateConstraint = 23,
    AtValidateConstraintRecurse = 24,
    AtAddIndexConstraint = 25,
    AtDropConstraint = 26,
    AtDropConstraintRecurse = 27,
    AtReAddComment = 28,
    AtAlterColumnType = 29,
    AtAlterColumnGenericOptions = 30,
    AtChangeOwner = 31,
    AtClusterOn = 32,
    AtDropCluster = 33,
    AtSetLogged = 34,
    AtSetUnLogged = 35,
    AtDropOids = 36,
    AtSetTableSpace = 37,
    AtSetRelOptions = 38,
    AtResetRelOptions = 39,
    AtReplaceRelOptions = 40,
    AtEnableTrig = 41,
    AtEnableAlwaysTrig = 42,
    AtEnableReplicaTrig = 43,
    AtDisableTrig = 44,
    AtEnableTrigAll = 45,
    AtDisableTrigAll = 46,
    AtEnableTrigUser = 47,
    AtDisableTrigUser = 48,
    AtEnableRule = 49,
    AtEnableAlwaysRule = 50,
    AtEnableReplicaRule = 51,
    AtDisableRule = 52,
    AtAddInherit = 53,
    AtDropInherit = 54,
    AtAddOf = 55,
    AtDropOf = 56,
    AtReplicaIdentity = 57,
    AtEnableRowSecurity = 58,
    AtDisableRowSecurity = 59,
    AtForceRowSecurity = 60,
    AtNoForceRowSecurity = 61,
    AtGenericOptions = 62,
    AtAttachPartition = 63,
    AtDetachPartition = 64,
    AtAddIdentity = 65,
    AtSetIdentity = 66,
    AtDropIdentity = 67,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum GrantTargetType {
    Undefined = 0,
    AclTargetObject = 1,
    AclTargetAllInSchema = 2,
    AclTargetDefaults = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum VariableSetKind {
    Undefined = 0,
    VarSetValue = 1,
    VarSetDefault = 2,
    VarSetCurrent = 3,
    VarSetMulti = 4,
    VarReset = 5,
    VarResetAll = 6,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum ConstrType {
    Undefined = 0,
    ConstrNull = 1,
    ConstrNotnull = 2,
    ConstrDefault = 3,
    ConstrIdentity = 4,
    ConstrGenerated = 5,
    ConstrCheck = 6,
    ConstrPrimary = 7,
    ConstrUnique = 8,
    ConstrExclusion = 9,
    ConstrForeign = 10,
    ConstrAttrDeferrable = 11,
    ConstrAttrNotDeferrable = 12,
    ConstrAttrDeferred = 13,
    ConstrAttrImmediate = 14,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum ImportForeignSchemaType {
    Undefined = 0,
    FdwImportSchemaAll = 1,
    FdwImportSchemaLimitTo = 2,
    FdwImportSchemaExcept = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum RoleStmtType {
    Undefined = 0,
    RolestmtRole = 1,
    RolestmtUser = 2,
    RolestmtGroup = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum FetchDirection {
    Undefined = 0,
    FetchForward = 1,
    FetchBackward = 2,
    FetchAbsolute = 3,
    FetchRelative = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum FunctionParameterMode {
    Undefined = 0,
    FuncParamIn = 1,
    FuncParamOut = 2,
    FuncParamInout = 3,
    FuncParamVariadic = 4,
    FuncParamTable = 5,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum TransactionStmtKind {
    Undefined = 0,
    TransStmtBegin = 1,
    TransStmtStart = 2,
    TransStmtCommit = 3,
    TransStmtRollback = 4,
    TransStmtSavepoint = 5,
    TransStmtRelease = 6,
    TransStmtRollbackTo = 7,
    TransStmtPrepare = 8,
    TransStmtCommitPrepared = 9,
    TransStmtRollbackPrepared = 10,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum ViewCheckOption {
    Undefined = 0,
    NoCheckOption = 1,
    LocalCheckOption = 2,
    CascadedCheckOption = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum ClusterOption {
    Undefined = 0,
    CluoptRecheck = 1,
    CluoptVerbose = 2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum DiscardMode {
    Undefined = 0,
    DiscardAll = 1,
    DiscardPlans = 2,
    DiscardSequences = 3,
    DiscardTemp = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum ReindexObjectType {
    Undefined = 0,
    ReindexObjectIndex = 1,
    ReindexObjectTable = 2,
    ReindexObjectSchema = 3,
    ReindexObjectSystem = 4,
    ReindexObjectDatabase = 5,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum AlterTsConfigType {
    AlterTsconfigTypeUndefined = 0,
    AlterTsconfigAddMapping = 1,
    AlterTsconfigAlterMappingForToken = 2,
    AlterTsconfigReplaceDict = 3,
    AlterTsconfigReplaceDictForToken = 4,
    AlterTsconfigDropMapping = 5,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum AlterSubscriptionType {
    Undefined = 0,
    AlterSubscriptionOptions = 1,
    AlterSubscriptionConnection = 2,
    AlterSubscriptionPublication = 3,
    AlterSubscriptionRefresh = 4,
    AlterSubscriptionEnabled = 5,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum OnCommitAction {
    Undefined = 0,
    OncommitNoop = 1,
    OncommitPreserveRows = 2,
    OncommitDeleteRows = 3,
    OncommitDrop = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum ParamKind {
    Undefined = 0,
    ParamExtern = 1,
    ParamExec = 2,
    ParamSublink = 3,
    ParamMultiexpr = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum CoercionContext {
    Undefined = 0,
    CoercionImplicit = 1,
    CoercionAssignment = 2,
    CoercionExplicit = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum CoercionForm {
    Undefined = 0,
    CoerceExplicitCall = 1,
    CoerceExplicitCast = 2,
    CoerceImplicitCast = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum BoolExprType {
    Undefined = 0,
    AndExpr = 1,
    OrExpr = 2,
    NotExpr = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum SubLinkType {
    Undefined = 0,
    ExistsSublink = 1,
    AllSublink = 2,
    AnySublink = 3,
    RowcompareSublink = 4,
    ExprSublink = 5,
    MultiexprSublink = 6,
    ArraySublink = 7,
    CteSublink = 8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum RowCompareType {
    Undefined = 0,
    RowcompareLt = 1,
    RowcompareLe = 2,
    RowcompareEq = 3,
    RowcompareGe = 4,
    RowcompareGt = 5,
    RowcompareNe = 6,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum MinMaxOp {
    Undefined = 0,
    IsGreatest = 1,
    IsLeast = 2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum SqlValueFunctionOp {
    SqlvalueFunctionOpUndefined = 0,
    SvfopCurrentDate = 1,
    SvfopCurrentTime = 2,
    SvfopCurrentTimeN = 3,
    SvfopCurrentTimestamp = 4,
    SvfopCurrentTimestampN = 5,
    SvfopLocaltime = 6,
    SvfopLocaltimeN = 7,
    SvfopLocaltimestamp = 8,
    SvfopLocaltimestampN = 9,
    SvfopCurrentRole = 10,
    SvfopCurrentUser = 11,
    SvfopUser = 12,
    SvfopSessionUser = 13,
    SvfopCurrentCatalog = 14,
    SvfopCurrentSchema = 15,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum XmlExprOp {
    Undefined = 0,
    IsXmlconcat = 1,
    IsXmlelement = 2,
    IsXmlforest = 3,
    IsXmlparse = 4,
    IsXmlpi = 5,
    IsXmlroot = 6,
    IsXmlserialize = 7,
    IsDocument = 8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum XmlOptionType {
    Undefined = 0,
    XmloptionDocument = 1,
    XmloptionContent = 2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum NullTestType {
    Undefined = 0,
    IsNull = 1,
    IsNotNull = 2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum BoolTestType {
    Undefined = 0,
    IsTrue = 1,
    IsNotTrue = 2,
    IsFalse = 3,
    IsNotFalse = 4,
    IsUnknown = 5,
    IsNotUnknown = 6,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum CmdType {
    Undefined = 0,
    CmdUnknown = 1,
    CmdSelect = 2,
    CmdUpdate = 3,
    CmdInsert = 4,
    CmdDelete = 5,
    CmdUtility = 6,
    CmdNothing = 7,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum JoinType {
    Undefined = 0,
    JoinInner = 1,
    JoinLeft = 2,
    JoinFull = 3,
    JoinRight = 4,
    JoinSemi = 5,
    JoinAnti = 6,
    JoinUniqueOuter = 7,
    JoinUniqueInner = 8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum AggStrategy {
    Undefined = 0,
    AggPlain = 1,
    AggSorted = 2,
    AggHashed = 3,
    AggMixed = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum AggSplit {
    Undefined = 0,
    AggsplitSimple = 1,
    AggsplitInitialSerial = 2,
    AggsplitFinalDeserial = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum SetOpCmd {
    Undefined = 0,
    SetopcmdIntersect = 1,
    SetopcmdIntersectAll = 2,
    SetopcmdExcept = 3,
    SetopcmdExceptAll = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum SetOpStrategy {
    Undefined = 0,
    SetopSorted = 1,
    SetopHashed = 2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum OnConflictAction {
    Undefined = 0,
    OnconflictNone = 1,
    OnconflictNothing = 2,
    OnconflictUpdate = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum LimitOption {
    Undefined = 0,
    Default = 1,
    Count = 2,
    WithTies = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum LockClauseStrength {
    Undefined = 0,
    LcsNone = 1,
    LcsForkeyshare = 2,
    LcsForshare = 3,
    LcsFornokeyupdate = 4,
    LcsForupdate = 5,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum LockWaitPolicy {
    Undefined = 0,
    LockWaitBlock = 1,
    LockWaitSkip = 2,
    LockWaitError = 3,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum LockTupleMode {
    Undefined = 0,
    LockTupleKeyShare = 1,
    LockTupleShare = 2,
    LockTupleNoKeyExclusive = 3,
    LockTupleExclusive = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum KeywordKind {
    NoKeyword = 0,
    UnreservedKeyword = 1,
    ColNameKeyword = 2,
    TypeFuncNameKeyword = 3,
    ReservedKeyword = 4,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum Token {
    Nul = 0,
    /// Single-character tokens that are returned 1:1 (identical with "self" list in scan.l)
    /// Either supporting syntax, or single-character operators (some can be both)
    /// Also see <https://www.postgresql.org/docs/12/sql-syntax-lexical.html#SQL-SYNTAX-SPECIAL-CHARS>
    ///
    /// "%"
    Ascii37 = 37,
    /// "("
    Ascii40 = 40,
    /// ")"
    Ascii41 = 41,
    /// "*"
    Ascii42 = 42,
    /// "+"
    Ascii43 = 43,
    /// ","
    Ascii44 = 44,
    /// "-"
    Ascii45 = 45,
    /// "."
    Ascii46 = 46,
    /// "/"
    Ascii47 = 47,
    /// ":"
    Ascii58 = 58,
    /// ";"
    Ascii59 = 59,
    /// "<"
    Ascii60 = 60,
    /// "="
    Ascii61 = 61,
    /// ">"
    Ascii62 = 62,
    /// "?"
    Ascii63 = 63,
    /// "["
    Ascii91 = 91,
    /// "\"
    Ascii92 = 92,
    /// "]"
    Ascii93 = 93,
    /// "^"
    Ascii94 = 94,
    /// Named tokens in scan.l
    Ident = 258,
    Uident = 259,
    Fconst = 260,
    Sconst = 261,
    Usconst = 262,
    Bconst = 263,
    Xconst = 264,
    Op = 265,
    Iconst = 266,
    Param = 267,
    Typecast = 268,
    DotDot = 269,
    ColonEquals = 270,
    EqualsGreater = 271,
    LessEquals = 272,
    GreaterEquals = 273,
    NotEquals = 274,
    SqlComment = 275,
    CComment = 276,
    AbortP = 277,
    AbsoluteP = 278,
    Access = 279,
    Action = 280,
    AddP = 281,
    Admin = 282,
    After = 283,
    Aggregate = 284,
    All = 285,
    Also = 286,
    Alter = 287,
    Always = 288,
    Analyse = 289,
    Analyze = 290,
    And = 291,
    Any = 292,
    Array = 293,
    As = 294,
    Asc = 295,
    Assertion = 296,
    Assignment = 297,
    Asymmetric = 298,
    At = 299,
    Attach = 300,
    Attribute = 301,
    Authorization = 302,
    Backward = 303,
    Before = 304,
    BeginP = 305,
    Between = 306,
    Bigint = 307,
    Binary = 308,
    Bit = 309,
    BooleanP = 310,
    Both = 311,
    By = 312,
    Cache = 313,
    Call = 314,
    Called = 315,
    Cascade = 316,
    Cascaded = 317,
    Case = 318,
    Cast = 319,
    CatalogP = 320,
    Chain = 321,
    CharP = 322,
    Character = 323,
    Characteristics = 324,
    Check = 325,
    Checkpoint = 326,
    Class = 327,
    Close = 328,
    Cluster = 329,
    Coalesce = 330,
    Collate = 331,
    Collation = 332,
    Column = 333,
    Columns = 334,
    Comment = 335,
    Comments = 336,
    Commit = 337,
    Committed = 338,
    Concurrently = 339,
    Configuration = 340,
    Conflict = 341,
    Connection = 342,
    Constraint = 343,
    Constraints = 344,
    ContentP = 345,
    ContinueP = 346,
    ConversionP = 347,
    Copy = 348,
    Cost = 349,
    Create = 350,
    Cross = 351,
    Csv = 352,
    Cube = 353,
    CurrentP = 354,
    CurrentCatalog = 355,
    CurrentDate = 356,
    CurrentRole = 357,
    CurrentSchema = 358,
    CurrentTime = 359,
    CurrentTimestamp = 360,
    CurrentUser = 361,
    Cursor = 362,
    Cycle = 363,
    DataP = 364,
    Database = 365,
    DayP = 366,
    Deallocate = 367,
    Dec = 368,
    DecimalP = 369,
    Declare = 370,
    Default = 371,
    Defaults = 372,
    Deferrable = 373,
    Deferred = 374,
    Definer = 375,
    DeleteP = 376,
    Delimiter = 377,
    Delimiters = 378,
    Depends = 379,
    Desc = 380,
    Detach = 381,
    Dictionary = 382,
    DisableP = 383,
    Discard = 384,
    Distinct = 385,
    Do = 386,
    DocumentP = 387,
    DomainP = 388,
    DoubleP = 389,
    Drop = 390,
    Each = 391,
    Else = 392,
    EnableP = 393,
    Encoding = 394,
    Encrypted = 395,
    EndP = 396,
    EnumP = 397,
    Escape = 398,
    Event = 399,
    Except = 400,
    Exclude = 401,
    Excluding = 402,
    Exclusive = 403,
    Execute = 404,
    Exists = 405,
    Explain = 406,
    Expression = 407,
    Extension = 408,
    External = 409,
    Extract = 410,
    FalseP = 411,
    Family = 412,
    Fetch = 413,
    Filter = 414,
    FirstP = 415,
    FloatP = 416,
    Following = 417,
    For = 418,
    Force = 419,
    Foreign = 420,
    Forward = 421,
    Freeze = 422,
    From = 423,
    Full = 424,
    Function = 425,
    Functions = 426,
    Generated = 427,
    Global = 428,
    Grant = 429,
    Granted = 430,
    Greatest = 431,
    GroupP = 432,
    Grouping = 433,
    Groups = 434,
    Handler = 435,
    Having = 436,
    HeaderP = 437,
    Hold = 438,
    HourP = 439,
    IdentityP = 440,
    IfP = 441,
    Ilike = 442,
    Immediate = 443,
    Immutable = 444,
    ImplicitP = 445,
    ImportP = 446,
    InP = 447,
    Include = 448,
    Including = 449,
    Increment = 450,
    Index = 451,
    Indexes = 452,
    Inherit = 453,
    Inherits = 454,
    Initially = 455,
    InlineP = 456,
    InnerP = 457,
    Inout = 458,
    InputP = 459,
    Insensitive = 460,
    Insert = 461,
    Instead = 462,
    IntP = 463,
    Integer = 464,
    Intersect = 465,
    Interval = 466,
    Into = 467,
    Invoker = 468,
    Is = 469,
    Isnull = 470,
    Isolation = 471,
    Join = 472,
    Key = 473,
    Label = 474,
    Language = 475,
    LargeP = 476,
    LastP = 477,
    LateralP = 478,
    Leading = 479,
    Leakproof = 480,
    Least = 481,
    Left = 482,
    Level = 483,
    Like = 484,
    Limit = 485,
    Listen = 486,
    Load = 487,
    Local = 488,
    Localtime = 489,
    Localtimestamp = 490,
    Location = 491,
    LockP = 492,
    Locked = 493,
    Logged = 494,
    Mapping = 495,
    Match = 496,
    Materialized = 497,
    Maxvalue = 498,
    Method = 499,
    MinuteP = 500,
    Minvalue = 501,
    Mode = 502,
    MonthP = 503,
    Move = 504,
    NameP = 505,
    Names = 506,
    National = 507,
    Natural = 508,
    Nchar = 509,
    New = 510,
    Next = 511,
    Nfc = 512,
    Nfd = 513,
    Nfkc = 514,
    Nfkd = 515,
    No = 516,
    None = 517,
    Normalize = 518,
    Normalized = 519,
    Not = 520,
    Nothing = 521,
    Notify = 522,
    Notnull = 523,
    Nowait = 524,
    NullP = 525,
    Nullif = 526,
    NullsP = 527,
    Numeric = 528,
    ObjectP = 529,
    Of = 530,
    Off = 531,
    Offset = 532,
    Oids = 533,
    Old = 534,
    On = 535,
    Only = 536,
    Operator = 537,
    Option = 538,
    Options = 539,
    Or = 540,
    Order = 541,
    Ordinality = 542,
    Others = 543,
    OutP = 544,
    OuterP = 545,
    Over = 546,
    Overlaps = 547,
    Overlay = 548,
    Overriding = 549,
    Owned = 550,
    Owner = 551,
    Parallel = 552,
    Parser = 553,
    Partial = 554,
    Partition = 555,
    Passing = 556,
    Password = 557,
    Placing = 558,
    Plans = 559,
    Policy = 560,
    Position = 561,
    Preceding = 562,
    Precision = 563,
    Preserve = 564,
    Prepare = 565,
    Prepared = 566,
    Primary = 567,
    Prior = 568,
    Privileges = 569,
    Procedural = 570,
    Procedure = 571,
    Procedures = 572,
    Program = 573,
    Publication = 574,
    Quote = 575,
    Range = 576,
    Read = 577,
    Real = 578,
    Reassign = 579,
    Recheck = 580,
    Recursive = 581,
    Ref = 582,
    References = 583,
    Referencing = 584,
    Refresh = 585,
    Reindex = 586,
    RelativeP = 587,
    Release = 588,
    Rename = 589,
    Repeatable = 590,
    Replace = 591,
    Replica = 592,
    Reset = 593,
    Restart = 594,
    Restrict = 595,
    Returning = 596,
    Returns = 597,
    Revoke = 598,
    Right = 599,
    Role = 600,
    Rollback = 601,
    Rollup = 602,
    Routine = 603,
    Routines = 604,
    Row = 605,
    Rows = 606,
    Rule = 607,
    Savepoint = 608,
    Schema = 609,
    Schemas = 610,
    Scroll = 611,
    Search = 612,
    SecondP = 613,
    Security = 614,
    Select = 615,
    Sequence = 616,
    Sequences = 617,
    Serializable = 618,
    Server = 619,
    Session = 620,
    SessionUser = 621,
    Set = 622,
    Sets = 623,
    Setof = 624,
    Share = 625,
    Show = 626,
    Similar = 627,
    Simple = 628,
    Skip = 629,
    Smallint = 630,
    Snapshot = 631,
    Some = 632,
    SqlP = 633,
    Stable = 634,
    StandaloneP = 635,
    Start = 636,
    Statement = 637,
    Statistics = 638,
    Stdin = 639,
    Stdout = 640,
    Storage = 641,
    Stored = 642,
    StrictP = 643,
    StripP = 644,
    Subscription = 645,
    Substring = 646,
    Support = 647,
    Symmetric = 648,
    Sysid = 649,
    SystemP = 650,
    Table = 651,
    Tables = 652,
    Tablesample = 653,
    Tablespace = 654,
    Temp = 655,
    Template = 656,
    Temporary = 657,
    TextP = 658,
    Then = 659,
    Ties = 660,
    Time = 661,
    Timestamp = 662,
    To = 663,
    Trailing = 664,
    Transaction = 665,
    Transform = 666,
    Treat = 667,
    Trigger = 668,
    Trim = 669,
    TrueP = 670,
    Truncate = 671,
    Trusted = 672,
    TypeP = 673,
    TypesP = 674,
    Uescape = 675,
    Unbounded = 676,
    Uncommitted = 677,
    Unencrypted = 678,
    Union = 679,
    Unique = 680,
    Unknown = 681,
    Unlisten = 682,
    Unlogged = 683,
    Until = 684,
    Update = 685,
    User = 686,
    Using = 687,
    Vacuum = 688,
    Valid = 689,
    Validate = 690,
    Validator = 691,
    ValueP = 692,
    Values = 693,
    Varchar = 694,
    Variadic = 695,
    Varying = 696,
    Verbose = 697,
    VersionP = 698,
    View = 699,
    Views = 700,
    Volatile = 701,
    When = 702,
    Where = 703,
    WhitespaceP = 704,
    Window = 705,
    With = 706,
    Within = 707,
    Without = 708,
    Work = 709,
    Wrapper = 710,
    Write = 711,
    XmlP = 712,
    Xmlattributes = 713,
    Xmlconcat = 714,
    Xmlelement = 715,
    Xmlexists = 716,
    Xmlforest = 717,
    Xmlnamespaces = 718,
    Xmlparse = 719,
    Xmlpi = 720,
    Xmlroot = 721,
    Xmlserialize = 722,
    Xmltable = 723,
    YearP = 724,
    YesP = 725,
    Zone = 726,
    NotLa = 727,
    NullsLa = 728,
    WithLa = 729,
    Postfixop = 730,
    Uminus = 731,
}
