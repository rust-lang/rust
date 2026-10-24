// exact-check
// ignore-order

const EXPECTED = [
    {
        'query': 'tyctxt, symbol -> bool',
        'others': [
            {
                'path': 'foo::TyCtxt',
                'name': 'has_attr',
                'displayType': "`TyCtxt`, Into<DefId>, `Symbol` -> `bool`",
            },
        ],
    },
    {
        'query': 'tyctxt, into<defid>, symbol -> bool',
        'others': [
            {
                'path': 'foo::TyCtxt',
                'name': 'has_attr',
                'displayType': "`TyCtxt`, `Into`<`DefId`>, `Symbol` -> `bool`",
            },
        ],
    },
    {
        'query': 'tyctxt, defid, symbol -> bool',
        'others': [
            {
                'path': 'foo::TyCtxt',
                'name': 'has_attr',
                'displayType': "`TyCtxt`, Into<`DefId`>, `Symbol` -> `bool`",
            },
        ],
    },
];
