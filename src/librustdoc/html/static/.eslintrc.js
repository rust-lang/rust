module.exports = {
    "env": {
        "browser": true,
        "es6": true
    },
    "extends": "eslint:recommended",
    "parserOptions": {
        "ecmaVersion": 2015,
        "sourceType": "module"
    },
    "rules": {
        "linebreak-style": [
            "error",
            "unix"
        ],
        "semi": [
            "error",
            "always"
        ],
        "quotes": [
            "error",
            "double"
        ],
        "linebreak-style": [
            "error",
            "unix"
        ],
        "no-trailing-spaces": "error",
        "no-var": ["error"],
        "prefer-const": ["error"],
        "prefer-arrow-callback": ["error"],
        "brace-style": [
            "error",
            "1tbs",
            { "allowSingleLine": false }
        ],
        "keyword-spacing": [
            "error",
            { "before": true, "after": true }
        ],
        "arrow-spacing": [
            "error",
            { "before": true, "after": true }
        ],
        "key-spacing": [
            "error",
            { "beforeColon": false, "afterColon": true, "mode": "strict" }
        ],
        "func-call-spacing": ["error", "never"],
    }
};
