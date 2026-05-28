// exact-check
// ignore-order
// ignore-tidy-linelength

const FILTER_CRATE = "search_method_disambiguate";

const EXPECTED = [
    {
        'query': 'MyTy -> bool',
        'others': [
            {
                'path': 'search_method_disambiguate::MyTy',
                'name': 'my_method',
                'href': '../search_method_disambiguate/struct.MyTy.html#impl-X-for-MyTy%3Cbool%3E/method.my_method'
            },
        ],
    },
    {
        'query': 'MyTy -> u8',
        'others': [
            {
                'path': 'search_method_disambiguate::MyTy',
                'name': 'my_method',
                'href': '../search_method_disambiguate/struct.MyTy.html#impl-X-for-MyTy%3Cu8%3E/method.my_method'
            },
        ],
    }
];
