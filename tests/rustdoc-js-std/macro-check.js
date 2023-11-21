// ignore-order

const EXPECTED = {
    'query': 'panic',
    'others': [
        { 'path': 'std', 'name': 'panic', ty: 16 }, // 16 is for macros
        { 'path': 'std', 'name': 'panic', ty: 2 }, // 2 is for modules
    ],
};
